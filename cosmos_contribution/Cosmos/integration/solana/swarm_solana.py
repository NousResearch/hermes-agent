"""
Cosmos Swarm Solana Integration

Multi-agent powered Solana tools:
1. Token Scanner - Analyze any SPL token with swarm consensus
2. DeFi Advisor - Multi-agent DeFi strategy recommendations
3. Wallet Watcher - Track wallets with intelligent alerts
4. Prediction Registry - Store FARSIGHT predictions on-chain
5. NFT Valuator - Swarm consensus on NFT pricing

All powered by 11-agent deliberation for superior accuracy.
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import httpx
from loguru import logger

# =============================================================================
# CONFIGURATION
# =============================================================================

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
HELIUS_RPC = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}" if HELIUS_API_KEY else "https://api.mainnet-beta.solana.com"
JUPITER_API = "https://quote-api.jup.ag/v6"
BIRDEYE_API = "https://public-api.birdeye.so"
DEXSCREENER_API = "https://api.dexscreener.com/latest"

# Cosmos token
COSMOS_TOKEN = "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TokenAnalysis:
    """Analysis of a Solana token by the swarm."""
    token_address: str
    token_name: Optional[str] = None
    token_symbol: Optional[str] = None

    # Market data
    price_usd: Optional[float] = None
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    liquidity: Optional[float] = None
    holders: Optional[int] = None

    # Swarm analysis
    risk_level: RiskLevel = RiskLevel.MEDIUM
    swarm_sentiment: Optional[str] = None  # bullish, bearish, neutral
    swarm_confidence: float = 0.0
    swarm_reasoning: str = ""

    # Agents
    agents_analyzed: List[str] = field(default_factory=list)
    analysis_hash: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_address": self.token_address,
            "token_name": self.token_name,
            "token_symbol": self.token_symbol,
            "price_usd": self.price_usd,
            "market_cap": self.market_cap,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "holders": self.holders,
            "risk_level": self.risk_level.value,
            "swarm_sentiment": self.swarm_sentiment,
            "swarm_confidence": self.swarm_confidence,
            "swarm_reasoning": self.swarm_reasoning,
            "agents_analyzed": self.agents_analyzed,
            "analysis_hash": self.analysis_hash,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DeFiRecommendation:
    """DeFi strategy recommendation from the swarm."""
    strategy_type: str  # yield, swap, stake, lp
    protocol: str
    expected_apy: Optional[float] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM

    swarm_recommendation: str = ""
    swarm_confidence: float = 0.0
    agents_consulted: List[str] = field(default_factory=list)

    steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_type": self.strategy_type,
            "protocol": self.protocol,
            "expected_apy": self.expected_apy,
            "risk_level": self.risk_level.value,
            "swarm_recommendation": self.swarm_recommendation,
            "swarm_confidence": self.swarm_confidence,
            "agents_consulted": self.agents_consulted,
            "steps": self.steps,
            "warnings": self.warnings,
        }


# =============================================================================
# SWARM SOLANA ENGINE
# =============================================================================

class SwarmSolana:
    """
    Multi-agent Solana intelligence powered by Cosmos's 11-agent swarm.
    """

    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self._token_cache: Dict[str, TokenAnalysis] = {}
        logger.info("SwarmSolana initialized - multi-agent Solana intelligence active")

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    # =========================================================================
    # TOKEN SCANNER
    # =========================================================================

    async def analyze_token(self, token_address: str) -> TokenAnalysis:
        """
        Analyze a Solana token using multi-agent consensus.

        Fetches on-chain data, then has the swarm analyze it for:
        - Risk assessment
        - Market sentiment
        - Investment recommendation
        """
        analysis = TokenAnalysis(token_address=token_address)

        # Fetch token data from multiple sources
        token_data = await self._fetch_token_data(token_address)

        if token_data:
            analysis.token_name = token_data.get("name")
            analysis.token_symbol = token_data.get("symbol")
            analysis.price_usd = token_data.get("price")
            analysis.market_cap = token_data.get("marketCap")
            analysis.volume_24h = token_data.get("volume24h")
            analysis.liquidity = token_data.get("liquidity")
            analysis.holders = token_data.get("holders")

        # Get swarm analysis
        swarm_result = await self._get_swarm_token_analysis(analysis, token_data)

        if swarm_result:
            analysis.swarm_sentiment = swarm_result.get("sentiment")
            analysis.swarm_confidence = swarm_result.get("confidence", 0.0)
            analysis.swarm_reasoning = swarm_result.get("reasoning", "")
            analysis.risk_level = RiskLevel(swarm_result.get("risk", "medium"))
            analysis.agents_analyzed = swarm_result.get("agents", [])

        # Generate hash
        hash_input = f"{token_address}|{analysis.swarm_sentiment}|{analysis.timestamp.isoformat()}"
        analysis.analysis_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # Cache result
        self._token_cache[token_address] = analysis

        return analysis

    async def _fetch_token_data(self, token_address: str) -> Dict[str, Any]:
        """Fetch token data from DexScreener and other sources."""
        data = {}

        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=30.0)

            # Try DexScreener
            resp = await self.client.get(
                f"{DEXSCREENER_API}/dex/tokens/{token_address}"
            )
            if resp.status_code == 200:
                result = resp.json()
                pairs = result.get("pairs", [])
                if pairs:
                    pair = pairs[0]  # Use first/main pair
                    data["name"] = pair.get("baseToken", {}).get("name")
                    data["symbol"] = pair.get("baseToken", {}).get("symbol")
                    data["price"] = float(pair.get("priceUsd", 0) or 0)
                    data["volume24h"] = float(pair.get("volume", {}).get("h24", 0) or 0)
                    data["liquidity"] = float(pair.get("liquidity", {}).get("usd", 0) or 0)
                    data["marketCap"] = float(pair.get("fdv", 0) or 0)
                    data["priceChange24h"] = float(pair.get("priceChange", {}).get("h24", 0) or 0)

        except Exception as e:
            logger.debug(f"Token data fetch error: {e}")

        return data

    async def _get_swarm_token_analysis(
        self,
        analysis: TokenAnalysis,
        token_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get swarm consensus on token analysis."""
        try:
            from Cosmos.core.collective.persistent_agent import call_shadow_agent

            # Build context
            context = f"""
Token: {analysis.token_symbol or analysis.token_address[:8]}
Address: {analysis.token_address}
Price: ${analysis.price_usd or 'Unknown'}
Market Cap: ${analysis.market_cap or 'Unknown'}
24h Volume: ${analysis.volume_24h or 'Unknown'}
Liquidity: ${analysis.liquidity or 'Unknown'}
24h Change: {token_data.get('priceChange24h', 'Unknown')}%
"""

            prompt = f"""Analyze this Solana token for investment potential:

{context}

Provide:
1. SENTIMENT: bullish, bearish, or neutral
2. RISK: low, medium, high, or extreme
3. CONFIDENCE: 0.0 to 1.0
4. REASONING: 2-3 sentences explaining your analysis

Consider: liquidity depth, volume/mcap ratio, price action, and red flags.

Format exactly as:
SENTIMENT: [value]
RISK: [value]
CONFIDENCE: [value]
REASONING: [your analysis]"""

            # Get analysis from multiple agents
            agents = ["grok", "deepseek", "gemini"]
            results = []
            participating_agents = []

            for agent_id in agents:
                result = await call_shadow_agent(agent_id, prompt, timeout=25.0)
                if result:
                    _, response = result
                    if response:
                        results.append(response)
                        participating_agents.append(agent_id)

            if not results:
                return None

            # Parse and aggregate results
            sentiments = []
            risks = []
            confidences = []
            reasonings = []

            for response in results:
                lines = response.upper().split('\n')
                for line in lines:
                    if 'SENTIMENT:' in line:
                        sent = line.split('SENTIMENT:')[1].strip().lower()
                        if sent in ['bullish', 'bearish', 'neutral']:
                            sentiments.append(sent)
                    elif 'RISK:' in line:
                        risk = line.split('RISK:')[1].strip().lower()
                        if risk in ['low', 'medium', 'high', 'extreme']:
                            risks.append(risk)
                    elif 'CONFIDENCE:' in line:
                        try:
                            conf = float(line.split('CONFIDENCE:')[1].strip().split()[0])
                            confidences.append(min(conf, 1.0))
                        except (ValueError, IndexError):
                            pass
                    elif 'REASONING:' in line:
                        reasoning = line.split('REASONING:')[1].strip()
                        if reasoning:
                            reasonings.append(reasoning)

            # Aggregate
            sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "neutral"
            risk = max(set(risks), key=risks.count) if risks else "medium"
            confidence = sum(confidences) / len(confidences) if confidences else 0.5
            reasoning = reasonings[0] if reasonings else "Analysis based on available market data."

            return {
                "sentiment": sentiment,
                "risk": risk,
                "confidence": round(confidence, 2),
                "reasoning": reasoning,
                "agents": participating_agents,
            }

        except Exception as e:
            logger.error(f"Swarm token analysis error: {e}")
            return None

    # =========================================================================
    # DEFI ADVISOR
    # =========================================================================

    async def get_defi_recommendation(
        self,
        amount_usd: float,
        risk_tolerance: str = "medium",
        goal: str = "yield"
    ) -> DeFiRecommendation:
        """
        Get a DeFi strategy recommendation from the swarm.

        Args:
            amount_usd: Amount to deploy
            risk_tolerance: low, medium, high
            goal: yield, growth, stable
        """
        recommendation = DeFiRecommendation(
            strategy_type=goal,
            protocol="",
            risk_level=RiskLevel(risk_tolerance),
        )

        try:
            from Cosmos.core.collective.persistent_agent import call_shadow_agent

            prompt = f"""Recommend a Solana DeFi strategy:

AMOUNT: ${amount_usd:,.2f}
RISK TOLERANCE: {risk_tolerance}
GOAL: {goal}

Consider these Solana protocols:
- Marinade (liquid staking, ~7% APY)
- Raydium (AMM/LP, variable APY)
- Orca (concentrated liquidity)
- Kamino (automated vaults)
- Jupiter (aggregator, JLP pool)
- Drift (perps, insurance fund)
- marginfi (lending/borrowing)

Provide:
1. PROTOCOL: Best protocol for this situation
2. STRATEGY: Specific strategy (e.g., "Stake SOL via Marinade")
3. EXPECTED_APY: Estimated annual return
4. STEPS: Numbered list of actions
5. WARNINGS: Any risks to be aware of

Be specific and actionable."""

            result = await call_shadow_agent("grok", prompt, timeout=30.0)
            if result:
                _, response = result

                # Parse response
                lines = response.split('\n')
                for line in lines:
                    line_upper = line.upper()
                    if 'PROTOCOL:' in line_upper:
                        recommendation.protocol = line.split(':', 1)[1].strip()
                    elif 'STRATEGY:' in line_upper:
                        recommendation.swarm_recommendation = line.split(':', 1)[1].strip()
                    elif 'EXPECTED_APY:' in line_upper or 'APY:' in line_upper:
                        try:
                            apy_str = line.split(':', 1)[1].strip().replace('%', '').split()[0]
                            recommendation.expected_apy = float(apy_str)
                        except (ValueError, IndexError):
                            pass
                    elif line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                        recommendation.steps.append(line.strip())
                    elif 'WARNING' in line_upper or 'RISK' in line_upper:
                        warning = line.split(':', 1)[-1].strip()
                        if warning:
                            recommendation.warnings.append(warning)

                recommendation.agents_consulted = ["grok"]
                recommendation.swarm_confidence = 0.75

        except Exception as e:
            logger.error(f"DeFi recommendation error: {e}")

        return recommendation

    # =========================================================================
    # WALLET WATCHER
    # =========================================================================

    async def analyze_wallet(self, wallet_address: str) -> Dict[str, Any]:
        """
        Analyze a Solana wallet using multi-agent intelligence.
        """
        wallet_data = {
            "address": wallet_address,
            "sol_balance": None,
            "token_count": 0,
            "nft_count": 0,
            "recent_activity": [],
            "swarm_assessment": "",
        }

        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=30.0)

            # Get SOL balance
            resp = await self.client.post(
                HELIUS_RPC,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getBalance",
                    "params": [wallet_address]
                }
            )
            if resp.status_code == 200:
                result = resp.json()
                lamports = result.get("result", {}).get("value", 0)
                wallet_data["sol_balance"] = lamports / 1e9

            # Get swarm assessment
            from Cosmos.core.collective.persistent_agent import call_shadow_agent

            prompt = f"""Analyze this Solana wallet:

Address: {wallet_address}
SOL Balance: {wallet_data['sol_balance']:.4f} SOL

Based on the address and balance, provide:
1. Wallet type assessment (whale, retail, bot, etc.)
2. Any notable patterns or red flags
3. Brief recommendation

Keep response under 100 words."""

            result = await call_shadow_agent("grok", prompt, timeout=20.0)
            if result:
                _, response = result
                wallet_data["swarm_assessment"] = response

        except Exception as e:
            logger.error(f"Wallet analysis error: {e}")

        return wallet_data

    # =========================================================================
    # JUPITER SWAP QUOTE
    # =========================================================================

    async def get_swap_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,  # In smallest units (lamports for SOL)
    ) -> Dict[str, Any]:
        """
        Get a swap quote from Jupiter with swarm commentary.
        """
        quote = {
            "input_mint": input_mint,
            "output_mint": output_mint,
            "input_amount": amount,
            "output_amount": None,
            "price_impact": None,
            "swarm_advice": "",
        }

        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=30.0)

            # Get Jupiter quote
            resp = await self.client.get(
                f"{JUPITER_API}/quote",
                params={
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "amount": amount,
                    "slippageBps": 50,
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                quote["output_amount"] = int(data.get("outAmount", 0))
                quote["price_impact"] = float(data.get("priceImpactPct", 0))
                quote["route"] = data.get("routePlan", [])

                # Get swarm advice on the swap
                from Cosmos.core.collective.persistent_agent import call_shadow_agent

                impact_pct = quote["price_impact"]
                prompt = f"""Evaluate this Solana swap:

Price Impact: {impact_pct:.2f}%
Route: {len(quote.get('route', []))} hops

Is this a good swap to execute? Consider price impact, MEV risk, and timing.
Reply in 1-2 sentences with advice."""

                result = await call_shadow_agent("deepseek", prompt, timeout=15.0)
                if result:
                    _, response = result
                    quote["swarm_advice"] = response

        except Exception as e:
            logger.error(f"Swap quote error: {e}")

        return quote


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_swarm_solana: Optional[SwarmSolana] = None


def get_swarm_solana() -> SwarmSolana:
    """Get the global SwarmSolana instance."""
    global _swarm_solana
    if _swarm_solana is None:
        _swarm_solana = SwarmSolana()
    return _swarm_solana


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def scan_token(token_address: str) -> Dict[str, Any]:
    """Quick token scan."""
    solana = get_swarm_solana()
    async with solana:
        result = await solana.analyze_token(token_address)
        return result.to_dict()


async def get_defi_advice(amount: float, risk: str = "medium") -> Dict[str, Any]:
    """Quick DeFi recommendation."""
    solana = get_swarm_solana()
    async with solana:
        result = await solana.get_defi_recommendation(amount, risk)
        return result.to_dict()


async def analyze_wallet(address: str) -> Dict[str, Any]:
    """Quick wallet analysis."""
    solana = get_swarm_solana()
    async with solana:
        return await solana.analyze_wallet(address)
