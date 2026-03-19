"""
Polymarket Prediction Engine

Uses the AI collective with 8 predictive markers to generate
high-confidence predictions on Polymarket events.

Predictions are generated every 5 minutes using:
1. Current odds momentum analysis
2. Volume/liquidity signals
3. Social sentiment (via Grok/Twitter)
4. News correlation (via web search)
5. Historical pattern matching
6. Related market correlation
7. Time-decay analysis
8. Multi-agent deliberation & voting

"The swarm sees what no single mind can." - Cosmos
"""

import asyncio
import json
import aiohttp
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import   Optional, Tuple
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger("polymarket_predictor")

# =============================================================================
# DATA CLASSES
# =============================================================================

class PredictionOutcome(Enum):
    PENDING = "pending"
    CORRECT = "correct"
    INCORRECT = "incorrect"
    EXPIRED = "expired"


@dataclass
class MarketSnapshot:
    """Snapshot of a Polymarket market."""
    market_id: str
    question: str
    outcomes: list[str]
    current_odds: dict[str, float]  # outcome -> probability
    volume_24h: float
    total_volume: float
    end_date: Optional[str]
    category: str
    url: str

    @classmethod
    def from_gamma_api(cls, data: dict) -> 'MarketSnapshot':
        """Parse from Gamma API response."""
        outcomes = []
        odds = {}

        # Handle different response formats
        if 'outcomes' in data:
            outcomes = data['outcomes']
        elif 'outcomePrices' in data:
            # Parse outcome prices
            prices = data.get('outcomePrices', '[]')
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except (json.JSONDecodeError, ValueError):
                    prices = []
            outcomes = ['Yes', 'No'] if len(prices) == 2 else [f"Option {i+1}" for i in range(len(prices))]
            for i, price in enumerate(prices):
                odds[outcomes[i]] = float(price) if price else 0.5

        if 'tokens' in data:
            for token in data.get('tokens', []):
                outcome = token.get('outcome', 'Unknown')
                price = float(token.get('price', 0.5))
                outcomes.append(outcome)
                odds[outcome] = price

        return cls(
            market_id=data.get('id', data.get('conditionId', '')),
            question=data.get('question', data.get('title', 'Unknown')),
            outcomes=outcomes or ['Yes', 'No'],
            current_odds=odds or {'Yes': 0.5, 'No': 0.5},
            volume_24h=float(data.get('volume24hr', 0)),
            total_volume=float(data.get('volume', data.get('volumeNum', 0))),
            end_date=data.get('endDate', data.get('endDateIso')),
            category=data.get('category', data.get('groupItemTitle', 'General')),
            url=f"https://polymarket.com/event/{data.get('slug', data.get('id', ''))}"
        )


@dataclass
class PredictiveSignal:
    """A single predictive signal/marker."""
    name: str
    signal_type: str  # momentum, sentiment, volume, news, correlation, time, consensus
    direction: str    # bullish, bearish, neutral
    confidence: float # 0-1
    reasoning: str
    data: dict = field(default_factory=dict)


@dataclass
class Prediction:
    """A prediction on a Polymarket outcome."""
    prediction_id: str
    market_id: str
    question: str
    predicted_outcome: str
    confidence: float  # 0-1
    current_odds: float
    signals: list[PredictiveSignal]
    reasoning: str
    agents_involved: list[str]
    created_at: str
    expires_at: str
    outcome: PredictionOutcome = PredictionOutcome.PENDING
    actual_result: Optional[str] = None
    resolved_at: Optional[str] = None

    def to_any(self) -> dict:
        d = asdict(self)
        d['outcome'] = self.outcome.value
        d['signals'] = [asdict(s) for s in self.signals]
        # Add direction for UI
        d['direction'] = self.predicted_outcome
        # Add current price for UI
        d['current_price'] = self.current_odds
        # Add top 3 signals for UI display
        top_signals = sorted(self.signals, key=lambda s: s.confidence, reverse=True)[:3]
        d['top_signals'] = [
            {"name": s.name, "weight": s.confidence, "reasoning": s.reasoning}
            for s in top_signals
        ]
        # Add timestamp
        d['timestamp'] = self.created_at
        return d

    @classmethod
    def from_any(cls, data: dict) -> 'Prediction':
        # Filter to only known fields
        known_fields = {
            'prediction_id', 'market_id', 'question', 'predicted_outcome',
            'confidence', 'current_odds', 'signals', 'reasoning',
            'agents_involved', 'created_at', 'expires_at', 'outcome',
            'actual_result', 'resolved_at'
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        filtered['outcome'] = PredictionOutcome(filtered.get('outcome', 'pending'))
        filtered['signals'] = [PredictiveSignal(**s) for s in filtered.get('signals', [])]
        return cls(**filtered)


@dataclass
class PredictionStats:
    """Accuracy statistics."""
    total_predictions: int = 0
    correct: int = 0
    incorrect: int = 0
    pending: int = 0
    accuracy: float = 0.0
    streak: int = 0  # Current win/loss streak
    best_streak: int = 0
    by_category: dict = field(default_factory=dict)
    by_confidence: dict = field(default_factory=dict)


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

class PolymarketPredictor:
    """
    AI-powered Polymarket prediction engine.

    Uses 8 predictive markers + collective deliberation to generate
    high-confidence picks.
    """

    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent.parent / "data" / "polymarket"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_file = self.data_dir / "predictions.json"
        self.stats_file = self.data_dir / "stats.json"

        self.predictions: list[Prediction] = []
        self.stats = PredictionStats()
        self._load_data()

        self._running = False
        self._task = None

        # Agent query functions (injected)
        self._agent_funcs: dict = {}

    def _load_data(self):
        """Load predictions and stats from disk."""
        if self.predictions_file.exists():
            try:
                with open(self.predictions_file, 'r') as f:
                    data = json.load(f)
                    self.predictions = [Prediction.from_any(p) for p in data.get('predictions', [])]
            except Exception as e:
                logger.error(f"Failed to load predictions: {e}")

        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    self.stats = PredictionStats(**data)
            except Exception as e:
                logger.error(f"Failed to load stats: {e}")

    def _save_data(self):
        """Save predictions and stats to disk."""
        try:
            with open(self.predictions_file, 'w') as f:
                json.dump({
                    'predictions': [p.to_any() for p in self.predictions[-500:]],  # Keep last 500
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)

            with open(self.stats_file, 'w') as f:
                json.dump(asdict(self.stats), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save data: {e}")

    def register_agent(self, agent_id: str, query_func):
        """Register an agent's query function."""
        self._agent_funcs[agent_id] = query_func

    # -------------------------------------------------------------------------
    # MARKET FETCHING
    # -------------------------------------------------------------------------

    async def fetch_active_markets(self, limit: int = 20) -> list[MarketSnapshot]:
        """Fetch active markets from Polymarket."""
        markets = []

        try:
            async with aiohttp.ClientSession() as session:
                # Fetch from Gamma API
                url = f"{self.GAMMA_API}/markets?active=true&closed=false&limit={limit}&order=volume24hr&ascending=false"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data:
                            try:
                                market = MarketSnapshot.from_gamma_api(item)
                                if market.question and market.current_odds:
                                    markets.append(market)
                            except Exception as e:
                                logger.debug(f"Failed to parse market: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")

        logger.info(f"Fetched {len(markets)} active markets")
        return markets

    async def get_market_history(self, market_id: str) -> dict:
        """Get price history for a market."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.GAMMA_API}/markets/{market_id}/prices"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            logger.debug(f"Failed to fetch market history: {e}")
        return {}

    # -------------------------------------------------------------------------
    # PREDICTIVE SIGNALS (8 MARKERS)
    # -------------------------------------------------------------------------

    async def _analyze_momentum(self, market: MarketSnapshot) -> PredictiveSignal:
        """Signal 1: Analyze price/odds momentum."""
        history = await self.get_market_history(market.market_id)

        direction = "neutral"
        confidence = 0.5
        reasoning = "Insufficient price history"

        if history and 'history' in history:
            prices = history['history'][-24:]  # Last 24 data points
            if len(prices) >= 2:
                start_price = prices[0].get('p', 0.5) if isinstance(prices[0]) else 0.5
                end_price = prices[-1].get('p', 0.5) if isinstance(prices[-1]) else 0.5
                change = end_price - start_price

                if change > 0.05:
                    direction = "bullish"
                    confidence = min(0.7 + abs(change), 0.9)
                    reasoning = f"Price momentum up {change*100:.1f}% recently"
                elif change < -0.05:
                    direction = "bearish"
                    confidence = min(0.7 + abs(change), 0.9)
                    reasoning = f"Price momentum down {abs(change)*100:.1f}% recently"
                else:
                    direction = "neutral"
                    confidence = 0.5
                    reasoning = f"Price stable (±{abs(change)*100:.1f}%)"

        return PredictiveSignal(
            name="Momentum Analysis",
            signal_type="momentum",
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            data={"history_points": len(history.get('history', []))}
        )

    async def _analyze_volume(self, market: MarketSnapshot) -> PredictiveSignal:
        """Signal 2: Analyze volume/liquidity signals."""
        volume_24h = market.volume_24h
        total_volume = market.total_volume

        # High volume = more reliable odds
        if volume_24h > 100000:
            confidence = 0.85
            reasoning = f"High 24h volume (${volume_24h:,.0f}) - odds likely accurate"
        elif volume_24h > 10000:
            confidence = 0.7
            reasoning = f"Moderate volume (${volume_24h:,.0f}) - decent liquidity"
        else:
            confidence = 0.5
            reasoning = f"Low volume (${volume_24h:,.0f}) - odds may be unreliable"

        # Direction based on whether smart money is moving
        direction = "neutral"
        if total_volume > 1000000:
            direction = "bullish" if market.current_odds.get('Yes', 0.5) > 0.6 else "bearish"

        return PredictiveSignal(
            name="Volume Analysis",
            signal_type="volume",
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            data={"volume_24h": volume_24h, "total_volume": total_volume}
        )

    async def _analyze_social_sentiment(self, market: MarketSnapshot) -> PredictiveSignal:
        """Signal 3: Analyze social media sentiment via agents."""
        direction = "neutral"
        confidence = 0.5
        reasoning = "Social sentiment analysis pending"

        # Query Grok for Twitter sentiment
        if 'Grok' in self._agent_funcs:
            try:
                query = f"What is the current Twitter/X sentiment about: {market.question}? Is sentiment positive, negative, or neutral? Just give a brief assessment."
                result = await self._agent_funcs['Grok'](query, 500)
                if result:
                    # Handle all return types: tuple, or string
                    if isinstance(result, tuple):
                        response = result[0]
                    elif isinstance(result):
                        response = result.get('content', str(result))
                    else:
                        response = result
                    response = str(response) if response else ""
                    response_lower = response.lower()

                    if 'positive' in response_lower or 'bullish' in response_lower or 'optimistic' in response_lower:
                        direction = "bullish"
                        confidence = 0.7
                    elif 'negative' in response_lower or 'bearish' in response_lower or 'pessimistic' in response_lower:
                        direction = "bearish"
                        confidence = 0.7
                    else:
                        direction = "neutral"
                        confidence = 0.55

                    reasoning = f"Social sentiment: {response[:100]}..." if response else "No sentiment data"
            except Exception as e:
                logger.debug(f"Social sentiment analysis failed: {e}")

        return PredictiveSignal(
            name="Social Sentiment",
            signal_type="sentiment",
            direction=direction,
            confidence=confidence,
            reasoning=reasoning
        )

    async def _analyze_news_correlation(self, market: MarketSnapshot) -> PredictiveSignal:
        """Signal 4: Analyze news correlation."""
        direction = "neutral"
        confidence = 0.5
        reasoning = "News analysis pending"

        # Use Gemini or Perplexity for news
        agent = 'Gemini' if 'Gemini' in self._agent_funcs else 'Perplexity' if 'Perplexity' in self._agent_funcs else None

        if agent and agent in self._agent_funcs:
            try:
                query = f"What are the latest news developments related to: {market.question}? How might recent news affect the outcome? Brief answer."
                result = await self._agent_funcs[agent](query, 500)
                if result:
                    # Handle all return types: tuple, or string
                    if isinstance(result, tuple):
                        response = result[0]
                    elif isinstance(result):
                        response = result.get('content', str(result))
                    else:
                        response = result
                    response = str(response) if response else ""
                    response_lower = response.lower()

                    # Parse sentiment from response
                    positive_words = ['positive', 'likely', 'expected', 'confirm', 'support', 'favor']
                    negative_words = ['negative', 'unlikely', 'doubt', 'against', 'oppose', 'fail']

                    pos_count = sum(1 for w in positive_words if w in response_lower)
                    neg_count = sum(1 for w in negative_words if w in response_lower)

                    if pos_count > neg_count:
                        direction = "bullish"
                        confidence = 0.65
                    elif neg_count > pos_count:
                        direction = "bearish"
                        confidence = 0.65

                    reasoning = f"News analysis: {response[:100]}..." if response else "No news data"
            except Exception as e:
                logger.debug(f"News analysis failed: {e}")

        return PredictiveSignal(
            name="News Correlation",
            signal_type="news",
            direction=direction,
            confidence=confidence,
            reasoning=reasoning
        )

    async def _analyze_historical_patterns(self, market: MarketSnapshot) -> PredictiveSignal:
        """Signal 5: Historical pattern matching."""
        # Check our past predictions for similar markets
        similar_preds = [p for p in self.predictions if p.outcome != PredictionOutcome.PENDING]

        direction = "neutral"
        confidence = 0.5
        reasoning = "No historical patterns found"

        if similar_preds:
            # Find similar categories/questions
            category_preds = [p for p in similar_preds if market.category.lower() in p.question.lower()]
            if category_preds:
                correct = sum(1 for p in category_preds if p.outcome == PredictionOutcome.CORRECT)
                total = len(category_preds)

                if total >= 3:
                    success_rate = correct / total
                    confidence = 0.5 + (success_rate - 0.5) * 0.3

                    if success_rate > 0.6:
                        direction = "bullish"  # We tend to be right on these
                        reasoning = f"Historical success in {market.category}: {correct}/{total} ({success_rate*100:.0f}%)"
                    elif success_rate < 0.4:
                        direction = "bearish"  # Inverse signal
                        reasoning = f"Historical contrarian signal: {correct}/{total} ({success_rate*100:.0f}%)"

        return PredictiveSignal(
            name="Historical Patterns",
            signal_type="correlation",
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            data={"similar_count": len(similar_preds)}
        )

    async def _analyze_related_markets(self, market: MarketSnapshot, all_markets: list[MarketSnapshot]) -> PredictiveSignal:
        """Signal 6: Related market correlation."""
        direction = "neutral"
        confidence = 0.5
        reasoning = "No related markets found"

        # Find related markets by keyword matching
        keywords = market.question.lower().split()[:5]
        related = []

        for other in all_markets:
            if other.market_id == market.market_id:
                continue
            other_lower = other.question.lower()
            matches = sum(1 for kw in keywords if kw in other_lower and len(kw) > 3)
            if matches >= 2:
                related.append(other)

        if related:
            # Check if related markets are moving in same direction
            bullish_count = sum(1 for r in related if r.current_odds.get('Yes', 0.5) > 0.55)
            bearish_count = sum(1 for r in related if r.current_odds.get('Yes', 0.5) < 0.45)

            if bullish_count > bearish_count:
                direction = "bullish"
                confidence = 0.6
                reasoning = f"Related markets leaning bullish ({bullish_count}/{len(related)})"
            elif bearish_count > bullish_count:
                direction = "bearish"
                confidence = 0.6
                reasoning = f"Related markets leaning bearish ({bearish_count}/{len(related)})"
            else:
                reasoning = f"Related markets mixed ({len(related)} found)"

        return PredictiveSignal(
            name="Related Markets",
            signal_type="correlation",
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            data={"related_count": len(related)}
        )

    async def _analyze_time_decay(self, market: MarketSnapshot) -> PredictiveSignal:
        """Signal 7: Time-to-resolution analysis."""
        direction = "neutral"
        confidence = 0.5
        reasoning = "No end date"

        if market.end_date:
            try:
                end = datetime.fromisoformat(market.end_date.replace('Z', '+00:00'))
                now = datetime.now(end.tzinfo) if end.tzinfo else datetime.now()
                days_left = (end - now).days

                if days_left < 1:
                    # Very close to resolution - current odds likely accurate
                    confidence = 0.85
                    current_leader = max(market.current_odds.items(), key=lambda x: x[1])
                    direction = "bullish" if current_leader[0] == 'Yes' else "bearish"
                    reasoning = f"Resolving soon ({days_left}d) - current odds likely final"
                elif days_left < 7:
                    confidence = 0.7
                    reasoning = f"Resolving in {days_left} days - limited time for movement"
                elif days_left > 30:
                    confidence = 0.5
                    reasoning = f"Long-dated ({days_left}d) - high uncertainty"
                else:
                    confidence = 0.6
                    reasoning = f"Medium-term ({days_left}d)"
            except Exception:
                pass

        return PredictiveSignal(
            name="Time Analysis",
            signal_type="time",
            direction=direction,
            confidence=confidence,
            reasoning=reasoning
        )

    async def _collective_deliberation(self, market: MarketSnapshot, signals: list[PredictiveSignal]) -> Tuple[PredictiveSignal, str]:
        """
        Signal 8: AGI-level Multi-Agent Deliberation and Research

        This is where the collective intelligence truly shines. Multiple agents:
        1. Research the topic deeply from their unique perspectives
        2. Share findings and challenge each other's reasoning
        3. Vote with weighted confidence based on expertise
        4. Synthesize into a final collective prediction

        Takes 1-2 minutes for thorough analysis.
        """
        logger.info(f"🧠 Starting AGI-level collective deliberation on: {market.question[:50]}...")
        logger.info(f"Available agents for deliberation: {list(self._agent_funcs.keys())}")

        agents_used = []
        agent_analyses = {}
        votes = {"Yes": 0, "No": 0}
        vote_weights = {"Yes": 0.0, "No": 0.0}
        full_reasoning = []

        # Phase 1: Deep Research (parallel agent queries)
        # Each agent researches from their specialized perspective
        research_prompts = {
            "Grok": f"""As a real-time information analyst with access to X/Twitter:
Research this Polymarket question: "{market.question}"

Current market odds: {json.dumps(market.current_odds)}
24h Volume: ${market.volume_24h:,.0f}

Tasks:
1. What is the current social media sentiment? Are people confident or uncertain?
2. dict recent viral posts or influential voices commenting on this?
3. What does crowd wisdom suggest?

Provide your analysis (200-300 words) then end with:
MY PREDICTION: YES or NO
CONFIDENCE: (1-10)""",

            "Gemini": f"""As a deep research analyst:
Analyze this prediction market question: "{market.question}"

Current market odds: {json.dumps(market.current_odds)}
24h Volume: ${market.volume_24h:,.0f}

Tasks:
1. What are the key facts and recent developments?
2. What are the strongest arguments for each outcome?
3. What information asymmetry might exist?

Provide detailed analysis (200-300 words) then end with:
MY PREDICTION: YES or NO
CONFIDENCE: (1-10)""",

            "DeepSeek": f"""As a logical reasoning specialist:
Evaluate this prediction: "{market.question}"

Current odds: {json.dumps(market.current_odds)}
Market activity: ${market.volume_24h:,.0f} volume

Analyze:
1. Base rates - historically, how often do events like this resolve YES?
2. Current odds reflect ${market.total_volume:,.0f} in total bets - are the crowds likely right?
3. What would need to happen for the underdog outcome?

Reasoning analysis (200-300 words) then end with:
MY PREDICTION: YES or NO
CONFIDENCE: (1-10)""",

            "Kimi": f"""As a thoughtful analyst with long-term perspective:
Analyze this prediction market: "{market.question}"

Current state:
- Odds: {json.dumps(market.current_odds)}
- Volume: ${market.volume_24h:,.0f} (24h) / ${market.total_volume:,.0f} total

Consider:
1. Long-term trends and historical context
2. Geopolitical or systemic factors that could affect this
3. What are the second-order effects people might be missing?

Thoughtful analysis (200-300 words) then end with:
MY PREDICTION: YES or NO
CONFIDENCE: (1-10)""",

            "Cosmos": f"""As the Swarm Mind orchestrator:
Final synthesis on: "{market.question}"

Market data:
- Current odds: {json.dumps(market.current_odds)}
- 24h Volume: ${market.volume_24h:,.0f}
- Total Volume: ${market.total_volume:,.0f}
- End Date: {market.end_date or 'Unknown'}

Your signals:
{chr(10).join([f"- {s.name}: {s.direction} ({s.confidence:.0%}) - {s.reasoning}" for s in signals])}

As the swarm coordinator, weigh all evidence and provide:
1. Key factors that will determine the outcome
2. Where do you see edge vs market consensus?
3. Risk factors that could invalidate the prediction

Analysis (200-300 words) then end with:
MY PREDICTION: YES or NO
CONFIDENCE: (1-10)""",
        }

        # Execute research queries in parallel
        logger.info("📚 Phase 1: Parallel deep research across agents...")
        research_tasks = []

        # Helper to avoid closure issue
        def make_query_task(aid, p, query_func):
            async def query_agent():
                try:
                    logger.debug(f"Querying agent {aid}...")
                    result = await asyncio.wait_for(
                        query_func(p, 800),  # More tokens for detailed analysis
                        timeout=45.0  # Longer timeout for deep research
                    )
                    logger.debug(f"Agent {aid} returned result")
                    return (aid, result[0] if isinstance(result, tuple) else result)
                except asyncio.TimeoutError:
                    logger.warning(f"Agent {aid} timed out")
                    return (aid, None)
                except Exception as e:
                    logger.debug(f"Research failed for {aid}: {e}")
                    return (aid, None)
            return query_agent()

        for agent_id, prompt in research_prompts.items():
            logger.debug(f"Checking agent {agent_id}: in funcs = {agent_id in self._agent_funcs}")
            if agent_id in self._agent_funcs:
                logger.info(f"Adding {agent_id} to research tasks")
                research_tasks.append(make_query_task(agent_id, prompt, self._agent_funcs[agent_id]))
            else:
                logger.warning(f"Agent {agent_id} not found in registered agents")

        # Gather all research results
        results = await asyncio.gather(*research_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple) and result[1]:
                agent_id, analysis = result
                agent_analyses[agent_id] = analysis
                agents_used.append(agent_id)
                logger.info(f"✓ {agent_id} completed research")

        # Phase 2: Parse votes and confidence from each analysis
        logger.info("🗳️ Phase 2: Extracting votes and confidence...")

        for agent_id, analysis in agent_analyses.items():
            if not analysis:
                continue

            # Handle different response types
            if isinstance(analysis):
                # If it's a  try to extract content
                analysis = analysis.get('content', analysis.get('message', analysis.get('text', str(analysis))))
            if not isinstance(analysis, str):
                analysis = str(analysis)

            analysis_upper = analysis.upper()

            # Extract prediction
            predicted_yes = False
            if "MY PREDICTION: YES" in analysis_upper or "MY PREDICTION:YES" in analysis_upper:
                predicted_yes = True
            elif "MY PREDICTION: NO" in analysis_upper or "MY PREDICTION:NO" in analysis_upper:
                predicted_yes = False
            else:
                # Fallback to first YES/NO mention
                yes_pos = analysis_upper.find("YES")
                no_pos = analysis_upper.find("NO")
                if yes_pos != -1 and (no_pos == -1 or yes_pos < no_pos):
                    predicted_yes = True

            # Extract confidence (1-10)
            confidence = 5  # Default
            import re
            conf_match = re.search(r'CONFIDENCE[:\s]*(\d+)', analysis_upper)
            if conf_match:
                confidence = min(10, max(1, int(conf_match.group(1))))

            # Weight vote by confidence
            weight = confidence / 10.0
            if predicted_yes:
                votes["Yes"] += 1
                vote_weights["Yes"] += weight
            else:
                votes["No"] += 1
                vote_weights["No"] += weight

            # Store reasoning
            prediction_str = "YES" if predicted_yes else "NO"
            full_reasoning.append(f"**{agent_id}** ({prediction_str}, conf={confidence}/10):\n{analysis[:500]}...")
            logger.info(f"  {agent_id}: {prediction_str} (confidence {confidence}/10)")

        # Phase 3: Weighted consensus calculation
        logger.info("⚖️ Phase 3: Calculating weighted consensus...")

        total_weight = vote_weights["Yes"] + vote_weights["No"]
        total_votes = votes["Yes"] + votes["No"]

        if total_weight > 0:
            yes_weighted_pct = vote_weights["Yes"] / total_weight

            # Strong consensus thresholds
            if yes_weighted_pct > 0.70:
                direction = "bullish"
                confidence = 0.70 + (yes_weighted_pct - 0.70) * 0.5  # 0.70-0.85 range
            elif yes_weighted_pct < 0.30:
                direction = "bearish"
                confidence = 0.70 + (0.30 - yes_weighted_pct) * 0.5
            elif yes_weighted_pct > 0.55:
                direction = "bullish"
                confidence = 0.55 + (yes_weighted_pct - 0.55) * 0.5  # 0.55-0.625 range
            elif yes_weighted_pct < 0.45:
                direction = "bearish"
                confidence = 0.55 + (0.45 - yes_weighted_pct) * 0.5
            else:
                # Very close - market efficient, no edge
                direction = "neutral"
                confidence = 0.50

            reasoning = (
                f"Collective Intelligence Verdict: {votes['Yes']} YES / {votes['No']} NO\n"
                f"Weighted Score: {yes_weighted_pct:.1%} YES confidence\n"
                f"Agents consulted: {', '.join(agents_used)}"
            )
        else:
            direction = "neutral"
            confidence = 0.5
            reasoning = "Insufficient agent consensus - no prediction edge"

        signal = PredictiveSignal(
            name="Collective Intelligence",
            signal_type="consensus",
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            data={
                "votes": votes,
                "vote_weights": vote_weights,
                "agents": agents_used,
                "weighted_yes_pct": yes_weighted_pct if total_weight > 0 else 0.5
            }
        )

        # Compile full deliberation transcript
        deliberation_transcript = (
            f"# Collective Deliberation on: {market.question}\n\n"
            f"## Market Context\n"
            f"- Odds: {json.dumps(market.current_odds)}\n"
            f"- Volume: ${market.volume_24h:,.0f} (24h) / ${market.total_volume:,.0f} (total)\n\n"
            f"## Agent Analyses\n\n" +
            "\n\n---\n\n".join(full_reasoning) +
            f"\n\n## Final Verdict\n{reasoning}"
        )

        logger.info(f"✅ Deliberation complete: {direction} ({confidence:.0%})")
        return signal, deliberation_transcript

    # -------------------------------------------------------------------------
    # PREDICTION GENERATION
    # -------------------------------------------------------------------------

    async def generate_predictions(self, count: int = 2) -> list[Prediction]:
        """Generate predictions using all 8 signals."""
        logger.info(f"Generating {count} predictions...")

        # Fetch markets
        markets = await self.fetch_active_markets(limit=30)
        if not markets:
            logger.warning("No markets fetched")
            return []

        # Score each market for prediction potential
        scored_markets = []

        for market in markets:
            # Skip very lopsided markets (>98% or <2%) - relaxed from 90%/10%
            max_odds = max(market.current_odds.values()) if market.current_odds else 0.5
            min_odds = min(market.current_odds.values()) if market.current_odds else 0.5

            if max_odds > 0.98 or min_odds < 0.02:
                logger.debug(f"Skipping lopsided market: {market.question[:40]}... ({max_odds:.2%})")
                continue

            # Score based on volume and odds - prefer higher volume, slightly favor balanced odds
            volume_score = min(market.volume_24h / 100000, 1.0)  # Scale to $100k

            # Still interesting if one side is favored, just need some uncertainty
            uncertainty_score = 1 - (abs(max_odds - 0.5) ** 2)  # Squared to be less punishing

            score = volume_score * 0.7 + uncertainty_score * 0.3
            scored_markets.append((market, score))
            logger.debug(f"Scored market: {market.question[:40]}... score={score:.2f} odds={max_odds:.2%}")

        # Sort by score and take top candidates
        scored_markets.sort(key=lambda x: x[1], reverse=True)
        candidates = [m for m, _ in scored_markets[:count * 2]]

        predictions = []

        logger.info(f"Analyzing {min(count, len(candidates))} candidate markets...")

        for market in candidates[:count]:
            try:
                prediction = await self._analyze_market(market, markets)
                if prediction:
                    if prediction.confidence >= 0.50:  # Accept dict prediction with some confidence
                        predictions.append(prediction)
                        self.predictions.append(prediction)
                        logger.info(f"Generated prediction: {prediction.predicted_outcome} on '{market.question[:50]}...' (conf={prediction.confidence:.2%})")
                    else:
                        logger.debug(f"Skipped low confidence prediction: {prediction.confidence:.2%}")
                else:
                    logger.debug(f"No prediction generated for: {market.question[:50]}...")
            except Exception as e:
                logger.error(f"Failed to analyze market {market.market_id}: {e}", exc_info=True)

        # Save
        if predictions:
            self._update_stats()
            self._save_data()

        logger.info(f"Generated {len(predictions)} predictions")
        return predictions

    async def _analyze_market(self, market: MarketSnapshot, all_markets: list[MarketSnapshot]) -> Optional[Prediction]:
        """Run full analysis on a single market."""
        logger.debug(f"Analyzing: {market.question[:50]}...")

        # Gather all 8 signals
        signals = await asyncio.gather(
            self._analyze_momentum(market),
            self._analyze_volume(market),
            self._analyze_social_sentiment(market),
            self._analyze_news_correlation(market),
            self._analyze_historical_patterns(market),
            self._analyze_related_markets(market, all_markets),
            self._analyze_time_decay(market),
        )
        signals = list(signals)

        # Signal 8: Collective deliberation
        consensus_signal, deliberation_reasoning = await self._collective_deliberation(market, signals)
        signals.append(consensus_signal)

        # Aggregate signals into final prediction
        bullish_weight = 0
        bearish_weight = 0

        for signal in signals:
            weight = signal.confidence
            if signal.signal_type == "consensus":
                weight *= 1.5  # Weight consensus higher

            if signal.direction == "bullish":
                bullish_weight += weight
            elif signal.direction == "bearish":
                bearish_weight += weight

        total_weight = bullish_weight + bearish_weight
        if total_weight == 0:
            return None

        # Determine prediction
        if bullish_weight > bearish_weight:
            predicted_outcome = "Yes"
            confidence = bullish_weight / (total_weight + 1)  # Normalize
        else:
            predicted_outcome = "No"
            confidence = bearish_weight / (total_weight + 1)

        # Cap confidence
        confidence = min(max(confidence, 0.5), 0.95)

        # Generate prediction ID
        pred_id = hashlib.sha256(
            f"{market.market_id}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Build reasoning
        reasoning_parts = [
            f"Prediction: {predicted_outcome} ({confidence:.0%} confidence)",
            "",
            "Signal Analysis:",
        ]
        for signal in signals:
            emoji = "🟢" if signal.direction == "bullish" else "🔴" if signal.direction == "bearish" else "⚪"
            reasoning_parts.append(f"{emoji} {signal.name}: {signal.reasoning}")

        reasoning_parts.append("")
        reasoning_parts.append(f"Deliberation: {deliberation_reasoning}")

        return Prediction(
            prediction_id=pred_id,
            market_id=market.market_id,
            question=market.question,
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            current_odds=market.current_odds.get(predicted_outcome, 0.5),
            signals=signals,
            reasoning="\n".join(reasoning_parts),
            agents_involved=consensus_signal.data.get('agents', []),
            created_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + timedelta(hours=24)).isoformat()
        )

    # -------------------------------------------------------------------------
    # ACCURACY TRACKING
    # -------------------------------------------------------------------------

    async def check_resolved_markets(self):
        """Check if dict pending predictions have resolved."""
        pending = [p for p in self.predictions if p.outcome == PredictionOutcome.PENDING]

        for prediction in pending:
            try:
                # Check if market resolved
                async with aiohttp.ClientSession() as session:
                    url = f"{self.GAMMA_API}/markets/{prediction.market_id}"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status == 200:
                            data = await resp.json()

                            if data.get('closed') or data.get('resolved'):
                                # Market resolved
                                resolution = data.get('resolution', data.get('outcome'))

                                if resolution:
                                    prediction.actual_result = resolution
                                    prediction.resolved_at = datetime.now().isoformat()

                                    # Check if we were correct
                                    if resolution.lower() == prediction.predicted_outcome.lower():
                                        prediction.outcome = PredictionOutcome.CORRECT
                                    else:
                                        prediction.outcome = PredictionOutcome.INCORRECT

                                    logger.info(f"Prediction {prediction.prediction_id} resolved: {prediction.outcome.value}")
            except Exception as e:
                logger.debug(f"Failed to check resolution for {prediction.prediction_id}: {e}")

        self._update_stats()
        self._save_data()

    def _update_stats(self):
        """Update accuracy statistics."""
        resolved = [p for p in self.predictions if p.outcome != PredictionOutcome.PENDING]

        self.stats.total_predictions = len(self.predictions)
        self.stats.correct = sum(1 for p in resolved if p.outcome == PredictionOutcome.CORRECT)
        self.stats.incorrect = sum(1 for p in resolved if p.outcome == PredictionOutcome.INCORRECT)
        self.stats.pending = sum(1 for p in self.predictions if p.outcome == PredictionOutcome.PENDING)

        if self.stats.correct + self.stats.incorrect > 0:
            self.stats.accuracy = self.stats.correct / (self.stats.correct + self.stats.incorrect)

        # Calculate streak
        streak = 0
        for p in reversed(resolved):
            if p.outcome == PredictionOutcome.CORRECT:
                streak += 1
            else:
                break

        self.stats.streak = streak
        self.stats.best_streak = max(self.stats.best_streak, streak)

    def get_recent_predictions(self, limit: int = 10) -> list[Prediction]:
        """Get most recent predictions."""
        return sorted(self.predictions, key=lambda p: p.created_at, reverse=True)[:limit]

    def get_stats(self) -> PredictionStats:
        """Get current accuracy stats."""
        return self.stats

    # -------------------------------------------------------------------------
    # SCHEDULER
    # -------------------------------------------------------------------------

    async def start(self, interval_minutes: int = 5):
        """Start the prediction scheduler."""
        if self._running:
            return

        self._running = True
        logger.info(f"Starting Polymarket Predictor (every {interval_minutes} min)")

        while self._running:
            try:
                # Generate new predictions
                await self.generate_predictions(count=2)

                # Check for resolved markets
                await self.check_resolved_markets()

            except Exception as e:
                logger.error(f"Prediction cycle failed: {e}")

            # Wait for next cycle
            await asyncio.sleep(interval_minutes * 60)

    def stop(self):
        """Stop the prediction scheduler."""
        self._running = False
        logger.info("Stopping Polymarket Predictor")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_predictor: Optional[PolymarketPredictor] = None


def get_predictor() -> PolymarketPredictor:
    """Get global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = PolymarketPredictor()
    return _predictor
