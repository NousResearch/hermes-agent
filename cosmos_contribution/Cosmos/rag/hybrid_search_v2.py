"""
cosmos Hybrid Search v2 - Intent Classification and Multi-Hop Retrieval

Q1 2025 Feature: Better Retrieval
- Query understanding with intent classification
- Multi-hop retrieval for complex questions
- Source attribution and confidence scoring
- Query expansion and reformulation
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Callable
from enum import Enum
from collections import defaultdict
import numpy as np

from loguru import logger


class QueryIntent(Enum):
    """Classified query intents."""
    FACTUAL = "factual"           # "What is X?"
    PROCEDURAL = "procedural"     # "How do I X?"
    COMPARATIVE = "comparative"   # "What's the difference between X and Y?"
    CAUSAL = "causal"             # "Why does X happen?"
    TEMPORAL = "temporal"         # "When did X happen?"
    EXPLORATORY = "exploratory"   # "Tell me about X"
    VERIFICATION = "verification" # "Is it true that X?"
    AGGREGATION = "aggregation"   # "List all X"
    NAVIGATION = "navigation"     # "Find X in Y"


@dataclass
class QueryAnalysis:
    """Analysis of a query."""
    original_query: str
    intent: QueryIntent
    confidence: float

    # Extracted components
    entities: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)  # time, location, etc.

    # Query reformulations
    expanded_queries: list[str] = field(default_factory=list)
    decomposed_queries: list[str] = field(default_factory=list)

    # Search parameters
    search_depth: int = 1  # Number of hops
    require_exact: bool = False
    temporal_filter: Optional[tuple[datetime, datetime]] = None


@dataclass
class SearchHop:
    """A single hop in multi-hop retrieval."""
    hop_number: int
    query: str
    results: list[dict]
    confidence: float
    reasoning: str = ""


@dataclass
class AttributedResult:
    """Search result with source attribution."""
    content: str
    score: float
    confidence: float

    # Attribution
    source_id: str
    source_type: str  # "archival", "conversation", "graph", "web"
    source_timestamp: Optional[datetime] = None
    source_context: Optional[str] = None

    # Multi-hop info
    hop_path: list[str] = field(default_factory=list)
    derivation_chain: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict = field(default_factory=dict)


@dataclass
class HybridSearchResult:
    """Complete result from hybrid search."""
    query_analysis: QueryAnalysis
    results: list[AttributedResult]
    hops: list[SearchHop]
    total_confidence: float
    explanation: str


class HybridSearchV2:
    """
    Advanced hybrid search with intent classification and multi-hop retrieval.

    Features:
    - Query intent classification
    - Automatic query expansion
    - Multi-hop reasoning for complex queries
    - Source attribution with confidence
    - Result explanation generation
    """

    def __init__(
        self,
        max_hops: int = 3,
        min_confidence: float = 0.3,
        expansion_count: int = 3,
    ):
        self.max_hops = max_hops
        self.min_confidence = min_confidence
        self.expansion_count = expansion_count

        # Search backends (set externally)
        self.archival_search_fn: Optional[Callable] = None
        self.conversation_search_fn: Optional[Callable] = None
        self.graph_search_fn: Optional[Callable] = None

        # Embedding and LLM functions
        self.embed_fn: Optional[Callable] = None
        self.llm_fn: Optional[Callable] = None

        # Intent patterns
        self._intent_patterns = {
            QueryIntent.FACTUAL: [
                r'^what (is|are|was|were)\b',
                r'^who (is|are|was|were)\b',
                r'^define\b',
                r'meaning of',
            ],
            QueryIntent.PROCEDURAL: [
                r'^how (do|can|to|should)\b',
                r'steps to',
                r'way to',
                r'process of',
            ],
            QueryIntent.COMPARATIVE: [
                r'difference between',
                r'compare',
                r'versus',
                r'vs\.?',
                r'better than',
                r'similar to',
            ],
            QueryIntent.CAUSAL: [
                r'^why\b',
                r'reason for',
                r'cause of',
                r'because',
            ],
            QueryIntent.TEMPORAL: [
                r'^when\b',
                r'date of',
                r'time of',
                r'since when',
                r'how long',
            ],
            QueryIntent.VERIFICATION: [
                r'^(is|are|was|were|do|does|did|can|could)\b.*\?$',
                r'true that',
                r'correct that',
            ],
            QueryIntent.AGGREGATION: [
                r'^list\b',
                r'^all\b',
                r'every',
                r'each',
                r'enumerate',
            ],
            QueryIntent.NAVIGATION: [
                r'^find\b',
                r'^locate\b',
                r'^where\b',
                r'search for',
            ],
        }

        self._lock = asyncio.Lock()

    async def search(
        self,
        query: str,
        max_results: int = 10,
        enable_multi_hop: bool = True,
        explain_results: bool = True,
    ) -> HybridSearchResult:
        """
        Perform advanced hybrid search.

        Args:
            query: The search query
            max_results: Maximum results to return
            enable_multi_hop: Enable multi-hop retrieval
            explain_results: Generate explanations

        Returns:
            Complete search result with attribution
        """
        async with self._lock:
            # Analyze query
            analysis = await self._analyze_query(query)

            # Expand query
            analysis.expanded_queries = await self._expand_query(query, analysis.intent)

            # Determine if multi-hop needed
            if enable_multi_hop and self._needs_multi_hop(analysis):
                analysis.decomposed_queries = await self._decompose_query(query, analysis)
                analysis.search_depth = min(len(analysis.decomposed_queries), self.max_hops)

            # Execute search
            hops = []
            all_results = []

            if analysis.search_depth > 1 and analysis.decomposed_queries:
                # Multi-hop search
                context = ""
                for i, sub_query in enumerate(analysis.decomposed_queries):
                    hop_query = f"{sub_query} {context}".strip()
                    hop_results = await self._search_single(hop_query, max_results)

                    hop = SearchHop(
                        hop_number=i + 1,
                        query=hop_query,
                        results=[r.metadata for r in hop_results],
                        confidence=np.mean([r.confidence for r in hop_results]) if hop_results else 0,
                        reasoning=f"Searching for: {sub_query}",
                    )
                    hops.append(hop)

                    # Accumulate context for next hop
                    if hop_results:
                        context = " ".join([r.content[:100] for r in hop_results[:2]])

                    # Update hop paths in results
                    for result in hop_results:
                        result.hop_path = [h.query for h in hops]
                        all_results.append(result)
            else:
                # Single search with expansions
                queries_to_search = [query] + analysis.expanded_queries[:self.expansion_count]

                for search_query in queries_to_search:
                    results = await self._search_single(search_query, max_results)
                    all_results.extend(results)

            # Deduplicate and rank
            unique_results = self._deduplicate_results(all_results)
            ranked_results = self._rank_results(unique_results, analysis)[:max_results]

            # Calculate confidence
            total_confidence = np.mean([r.confidence for r in ranked_results]) if ranked_results else 0

            # Generate explanation
            explanation = ""
            if explain_results:
                explanation = await self._generate_explanation(analysis, ranked_results, hops)

            return HybridSearchResult(
                query_analysis=analysis,
                results=ranked_results,
                hops=hops,
                total_confidence=float(total_confidence),
                explanation=explanation,
            )

    async def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine intent and extract components."""
        query_lower = query.lower().strip()

        # Classify intent
        intent = QueryIntent.EXPLORATORY  # Default
        confidence = 0.5

        for intent_type, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intent = intent_type
                    confidence = 0.8
                    break
            if confidence > 0.5:
                break

        # Extract entities (simple approach)
        entities = self._extract_entities(query)

        # Extract relationships
        relationships = self._extract_relationships(query)

        # Extract constraints
        constraints = self._extract_constraints(query)

        return QueryAnalysis(
            original_query=query,
            intent=intent,
            confidence=confidence,
            entities=entities,
            relationships=relationships,
            constraints=constraints,
        )

    def _extract_entities(self, query: str) -> list[str]:
        """Extract potential entities from query."""
        entities = []

        # Capitalized words (excluding first word)
        words = query.split()
        for i, word in enumerate(words[1:], 1):
            if word[0].isupper() and word.isalpha():
                entities.append(word)

        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        # Code-like patterns
        code = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', query)
        entities.extend(code)

        return list(set(entities))

    def _extract_relationships(self, query: str) -> list[str]:
        """Extract relationship mentions from query."""
        relationships = []

        rel_patterns = [
            r'(\w+)\s+(is|are|was|were)\s+(\w+)',
            r'(\w+)\s+(uses|used|using)\s+(\w+)',
            r'(\w+)\s+(has|have|had)\s+(\w+)',
            r'(\w+)\s+(and|or)\s+(\w+)',
        ]

        for pattern in rel_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                relationships.append(" ".join(match))

        return relationships

    def _extract_constraints(self, query: str) -> dict:
        """Extract constraints like time, location, etc."""
        constraints = {}

        # Time constraints
        time_patterns = [
            (r'(before|after|since|until)\s+(\d{4}|\w+\s+\d{4})', 'time'),
            (r'(yesterday|today|last\s+\w+|this\s+\w+)', 'time_relative'),
            (r'(in|during|at)\s+(\d{4})', 'year'),
        ]

        for pattern, constraint_type in time_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                constraints[constraint_type] = match.group()

        # Count constraints
        count_match = re.search(r'(top|first|last)\s+(\d+)', query, re.IGNORECASE)
        if count_match:
            constraints['limit'] = int(count_match.group(2))

        return constraints

    async def _expand_query(self, query: str, intent: QueryIntent) -> list[str]:
        """Expand query with synonyms and reformulations."""
        expansions = []

        if self.llm_fn:
            # Use LLM for smart expansion
            prompt = f"""Generate 3 alternative phrasings for this search query.
Query: {query}
Intent: {intent.value}

Return only the 3 alternatives, one per line."""

            try:
                result = await self.llm_fn(prompt)
                if result:
                    expansions = [line.strip() for line in result.strip().split('\n') if line.strip()]
            except Exception as e:
                logger.error(f"Query expansion failed: {e}")

        # Fallback: simple expansions
        if not expansions:
            # Remove question words
            simplified = re.sub(r'^(what|how|why|when|where|who|is|are|was|were)\s+', '', query.lower())
            if simplified != query.lower():
                expansions.append(simplified)

            # Add wildcards
            words = query.split()
            if len(words) >= 2:
                expansions.append(f"{words[0]} * {words[-1]}")

        return expansions[:self.expansion_count]

    def _needs_multi_hop(self, analysis: QueryAnalysis) -> bool:
        """Determine if query needs multi-hop retrieval."""
        # Complex intents often need multi-hop
        complex_intents = {QueryIntent.CAUSAL, QueryIntent.COMPARATIVE, QueryIntent.PROCEDURAL}
        if analysis.intent in complex_intents:
            return True

        # Multiple entities suggest relationship queries
        if len(analysis.entities) >= 2:
            return True

        # Explicit relationship mentions
        if analysis.relationships:
            return True

        return False

    async def _decompose_query(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> list[str]:
        """Decompose complex query into sub-queries."""
        if self.llm_fn:
            prompt = f"""Decompose this complex question into simpler sub-questions that can be answered sequentially.
Question: {query}
Intent: {analysis.intent.value}
Entities: {', '.join(analysis.entities)}

Return 2-3 sub-questions, one per line, that when answered together address the main question."""

            try:
                result = await self.llm_fn(prompt)
                if result:
                    sub_queries = [line.strip() for line in result.strip().split('\n') if line.strip()]
                    return sub_queries[:self.max_hops]
            except Exception as e:
                logger.error(f"Query decomposition failed: {e}")

        # Fallback: entity-based decomposition
        sub_queries = []
        for entity in analysis.entities[:self.max_hops]:
            sub_queries.append(f"What is {entity}?")

        if len(analysis.entities) >= 2:
            sub_queries.append(f"How are {' and '.join(analysis.entities[:2])} related?")

        return sub_queries or [query]

    async def _search_single(
        self,
        query: str,
        max_results: int,
    ) -> list[AttributedResult]:
        """Execute search across all backends."""
        results = []

        # Search archival memory
        if self.archival_search_fn:
            try:
                archival_results = await self.archival_search_fn(query, top_k=max_results)
                for r in archival_results:
                    results.append(AttributedResult(
                        content=r.get("content", ""),
                        score=r.get("score", 0.5),
                        confidence=self._calculate_confidence(r.get("score", 0.5), "archival"),
                        source_id=r.get("id", ""),
                        source_type="archival",
                        source_timestamp=datetime.fromisoformat(r["created_at"]) if r.get("created_at") else None,
                        metadata=r,
                    ))
            except Exception as e:
                logger.error(f"Archival search failed: {e}")

        # Search conversations
        if self.conversation_search_fn:
            try:
                conv_results = await self.conversation_search_fn(query, top_k=max_results)
                for r in conv_results:
                    results.append(AttributedResult(
                        content=r.get("content", ""),
                        score=r.get("score", 0.5),
                        confidence=self._calculate_confidence(r.get("score", 0.5), "conversation"),
                        source_id=r.get("id", ""),
                        source_type="conversation",
                        source_timestamp=datetime.fromisoformat(r["timestamp"]) if r.get("timestamp") else None,
                        source_context=r.get("context", ""),
                        metadata=r,
                    ))
            except Exception as e:
                logger.error(f"Conversation search failed: {e}")

        # Search knowledge graph
        if self.graph_search_fn:
            try:
                graph_results = await self.graph_search_fn(query, max_entities=max_results)
                default_score = graph_results.get("score", 0.5)
                for entity in graph_results.get("entities", []):
                    # Use per-entity score if available, otherwise use default
                    entity_score = entity.get("score", entity.get("relevance", default_score))
                    results.append(AttributedResult(
                        content=f"{entity.get('name', '')} ({entity.get('type', '')}): {entity.get('description', '')}",
                        score=entity_score,
                        confidence=self._calculate_confidence(entity_score, "graph"),
                        source_id=entity.get("id", ""),
                        source_type="graph",
                        metadata=entity,
                    ))
            except Exception as e:
                logger.error(f"Graph search failed: {e}")

        return results

    def _calculate_confidence(self, score: float, source_type: str) -> float:
        """Calculate confidence score based on source and score."""
        # Base confidence from score
        confidence = score

        # Adjust by source reliability
        source_weights = {
            "archival": 1.0,
            "conversation": 0.8,
            "graph": 0.9,
            "web": 0.6,
        }
        confidence *= source_weights.get(source_type, 0.7)

        return min(1.0, max(0.0, confidence))

    def _deduplicate_results(self, results: list[AttributedResult]) -> list[AttributedResult]:
        """Remove duplicate results."""
        seen_content = {}
        unique = []

        for result in results:
            # Hash content for dedup
            content_key = result.content[:200].lower()
            if content_key in seen_content:
                # Keep higher confidence one
                existing = seen_content[content_key]
                if result.confidence > existing.confidence:
                    unique.remove(existing)
                    unique.append(result)
                    seen_content[content_key] = result
            else:
                unique.append(result)
                seen_content[content_key] = result

        return unique

    def _rank_results(
        self,
        results: list[AttributedResult],
        analysis: QueryAnalysis,
    ) -> list[AttributedResult]:
        """Rank results based on relevance and confidence."""
        def score_result(r: AttributedResult) -> float:
            score = r.confidence

            # Boost if entities match
            content_lower = r.content.lower()
            for entity in analysis.entities:
                if entity.lower() in content_lower:
                    score *= 1.2

            # Boost by recency for temporal queries
            if analysis.intent == QueryIntent.TEMPORAL and r.source_timestamp:
                days_ago = (datetime.now() - r.source_timestamp).days
                recency_boost = 1.0 / (1.0 + days_ago / 365)
                score *= (0.8 + 0.2 * recency_boost)

            # Boost archival for factual queries
            if analysis.intent == QueryIntent.FACTUAL and r.source_type == "archival":
                score *= 1.1

            # Boost graph for comparative queries
            if analysis.intent == QueryIntent.COMPARATIVE and r.source_type == "graph":
                score *= 1.15

            return score

        return sorted(results, key=score_result, reverse=True)

    async def _generate_explanation(
        self,
        analysis: QueryAnalysis,
        results: list[AttributedResult],
        hops: list[SearchHop],
    ) -> str:
        """Generate explanation of search results."""
        parts = []

        # Query understanding
        parts.append(f"Query Type: {analysis.intent.value}")
        if analysis.entities:
            parts.append(f"Key Entities: {', '.join(analysis.entities)}")

        # Multi-hop explanation
        if hops:
            parts.append(f"Search Path ({len(hops)} hops):")
            for hop in hops:
                parts.append(f"  {hop.hop_number}. {hop.query} (confidence: {hop.confidence:.2f})")

        # Source breakdown
        source_counts = defaultdict(int)
        for r in results:
            source_counts[r.source_type] += 1

        if source_counts:
            parts.append("Sources: " + ", ".join(f"{k}: {v}" for k, v in source_counts.items()))

        # Confidence summary
        if results:
            avg_conf = np.mean([r.confidence for r in results])
            parts.append(f"Average Confidence: {avg_conf:.2f}")

        return "\n".join(parts)

    def get_stats(self) -> dict:
        """Get search system statistics."""
        return {
            "max_hops": self.max_hops,
            "min_confidence": self.min_confidence,
            "expansion_count": self.expansion_count,
            "has_archival_search": self.archival_search_fn is not None,
            "has_conversation_search": self.conversation_search_fn is not None,
            "has_graph_search": self.graph_search_fn is not None,
            "has_llm": self.llm_fn is not None,
        }
