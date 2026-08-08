#!/usr/bin/env python3
"""
Hybrid Retriever for Hermes Memory Engine

Combines:
1. Lexical search (BM25)
2. Semantic search (embeddings)
3. Temporal ranking (recency)

Integrates with:
- session_search() for cross-session queries
- memory_engine.py for vector DB operations
- mcp_memory for fact updates

Usage:
    retriever = HybridRetriever()
    results = retriever.search("authentication", top_k=5)
    results = retriever.search_with_context("authentication", include_window=True)
"""

import json
import math
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import subprocess
from pathlib import Path
from embedder import Embedder


@dataclass
class RankedResult:
    """Ranked search result with detailed scoring"""
    fact_id: str
    content: str
    source: str
    source_path: Optional[str] = None
    
    # Scoring components
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    temporal_score: float = 0.0
    reference_score: float = 0.0
    
    # Combined score
    combined_score: float = field(init=False)
    
    # Metadata
    freshness_tier: str = "medium"
    decay_weight: float = 1.0
    referenced_count: int = 0
    last_referenced: Optional[str] = None
    created_at: str = ""
    confidence: float = 0.9
    
    # Context
    neighboring_facts: List[str] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        # Calculate combined score with weights
        weights = {
            'bm25': 0.4,
            'semantic': 0.4,
            'temporal': 0.15,
            'reference': 0.05
        }
        
        self.combined_score = (
            self.bm25_score * weights['bm25'] +
            self.semantic_score * weights['semantic'] +
            self.temporal_score * weights['temporal'] +
            self.reference_score * weights['reference']
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'fact_id': self.fact_id,
            'content': self.content,
            'source': self.source,
            'source_path': self.source_path,
            'scores': {
                'bm25': round(self.bm25_score, 3),
                'semantic': round(self.semantic_score, 3),
                'temporal': round(self.temporal_score, 3),
                'reference': round(self.reference_score, 3),
                'combined': round(self.combined_score, 3)
            },
            'metadata': {
                'freshness_tier': self.freshness_tier,
                'decay_weight': round(self.decay_weight, 3),
                'referenced_count': self.referenced_count,
                'last_referenced': self.last_referenced,
                'created_at': self.created_at,
                'confidence': self.confidence
            },
            'context': {
                'neighboring_facts': self.neighboring_facts,
                'relationships': self.relationships
            }
        }


class HybridRetriever:
    """Main retriever combining multiple ranking signals"""
    
    def __init__(self, engine_path: Path = None):
        self.engine_path = engine_path or Path.home() / ".hermes/memory-engine"
        self.db_path = self.engine_path / "db/memory.db"
        self.config_path = self.engine_path / "config/memory-engine.yaml"
        
    def _call_memory_engine(self, args: List[str]) -> str:
        """Call memory_engine.py script"""
        script = self.engine_path / "scripts/memory_engine.py"
        result = subprocess.run(
            ["python3", str(script)] + args,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout
    
    def _bm25_score(self, query: str, text: str) -> float:
        """
        BM25-like lexical relevance scoring.
        
        Parameters:
        - k1: controls term frequency saturation (default: 1.5)
        - b: controls length normalization (default: 0.75)
        """
        k1 = 1.5
        b = 0.75
        
        query_terms = query.lower().split()
        text_lower = text.lower()
        
        score = 0.0
        doc_length = len(text_lower.split())
        avg_length = 100  # approximate average
        
        for term in query_terms:
            # Term frequency in document
            tf = text_lower.count(term)
            
            # IDF (inverse document frequency) - simplified
            # In real implementation, would use corpus statistics
            idf = math.log(2.0 / (1.0 + (tf / doc_length)))
            
            # Length normalization
            norm_length = 1.0 - b + b * (doc_length / avg_length)
            
            # BM25 formula
            score += idf * ((tf * (k1 + 1)) / (tf + k1 * norm_length))
        
        # Normalize to 0-1 range
        max_score = len(query_terms) * 10
        return min(1.0, score / max_score)
    
    def _calculate_temporal_score(self, created_at: str) -> float:
        """
        Calculate temporal relevance score based on age.
        
        Recent (< 7 days):  1.0 * 2.0 = 2.0 (boosted)
        Medium (7-30 days): 0.7 * 1.0 = 0.7
        Old (30-90 days):   0.3 * 0.5 = 0.15
        Archive (>90 days): 0.05 * 0.1 = 0.005
        """
        created = datetime.fromisoformat(created_at)
        days_old = (datetime.now() - created).days
        
        # Exponential decay
        decay_lambda = 0.05
        raw_decay = math.exp(-decay_lambda * days_old)
        
        # Freshness tier boost
        if days_old <= 7:
            boost = 2.0
        elif days_old <= 30:
            boost = 1.0
        elif days_old <= 90:
            boost = 0.5
        else:
            boost = 0.1
        
        return min(1.0, raw_decay * boost)
    
    def _calculate_reference_score(self, referenced_count: int, 
                                   last_referenced: Optional[str]) -> float:
        """
        Score based on how often and recently this fact was used.
        """
        # Reference frequency
        freq_score = min(1.0, referenced_count / 10.0)  # max 10 references
        
        # Recency of reference
        if last_referenced:
            ref_date = datetime.fromisoformat(last_referenced)
            days_since = (datetime.now() - ref_date).days
            recency_score = math.exp(-0.1 * days_since)
        else:
            recency_score = 0.0
        
        # Combined
        return (freq_score * 0.6) + (recency_score * 0.4)
    
    def _get_semantic_score(self, query: str, content: str) -> float:
        """
        Real semantic similarity using embeddings.
        
        Uses embedder.py to:
        1. Embed query
        2. Embed content
        3. Calculate cosine similarity
        """
        if not hasattr(self, '_embedder'):
            self._embedder = Embedder()
            self._embedder.connect()
        
        try:
            query_emb = self._embedder.embed_text(query)
            content_emb = self._embedder.embed_text(content)
            
            similarity = Embedder.cosine_similarity(query_emb, content_emb)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception:
            # Fallback: keyword overlap
            query_terms = set(query.lower().split())
            content_terms = set(content.lower().split())
            
            if not query_terms or not content_terms:
                return 0.0
            
            overlap = len(query_terms & content_terms)
            return overlap / len(query_terms)
    
    def search(self, query: str, top_k: int = 5,
              min_score: float = 0.3) -> List[RankedResult]:
        """
        Hybrid search combining BM25 + semantic + temporal signals.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            min_score: Minimum combined score threshold
        
        Returns:
            List of ranked results sorted by combined score
        """
        # For MVP, call memory_engine with query
        output = self._call_memory_engine(['--query', query])
        
        results = []
        
        # Parse output (simplified for demo)
        # In production, would parse structured JSON from memory_engine
        lines = output.strip().split('\n')
        
        # TODO: Full implementation with actual DB queries
        
        return results[:top_k]
    
    def search_with_context(self, query: str, top_k: int = 5,
                           window_size: int = 3) -> List[Dict]:
        """
        Search and include contextual window (neighboring sessions/facts).
        
        Args:
            query: Search query
            top_k: Number of primary results
            window_size: Sessions to fetch before/after
        
        Returns:
            Results with neighboring context
        """
        # Get primary results
        results = self.search(query, top_k)
        
        # For each result, fetch contextual window
        with_context = []
        
        for result in results:
            context = self._fetch_context_window(result.fact_id, window_size)
            result.neighboring_facts = context
            with_context.append(result.to_dict())
        
        return with_context
    
    def _fetch_context_window(self, fact_id: str, window_size: int) -> List[str]:
        """Fetch N neighboring facts/sessions"""
        # TODO: Query DB for related facts
        return []
    
    def search_json(self, query: str, top_k: int = 5) -> str:
        """Search and return JSON results"""
        results = self.search(query, top_k)
        return json.dumps([r.to_dict() for r in results], indent=2)


# ─────────────────────────────────────────────────────────────────
# INTEGRATION WITH MCP MEMORY
# ─────────────────────────────────────────────────────────────────

def integrate_with_mcp():
    """
    Integration point for mcp_memory tool.
    
    This function would be called by the MCP bridge to:
    1. Intercept memory_note tags
    2. Extract facts
    3. Index them with embeddings
    4. Store in vector DB
    """
    pass


# ─────────────────────────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Hybrid Retriever CLI")
        print("Usage: python hybrid_retriever.py <query>")
        sys.exit(1)
    
    query = ' '.join(sys.argv[1:])
    retriever = HybridRetriever()
    
    print(f"\n╔═══════════════════════════════════════════════════════╗")
    print(f"║ HYBRID RETRIEVAL: '{query}'")
    print(f"╚═══════════════════════════════════════════════════════╝\n")
    
    results = retriever.search(query, top_k=5)
    
    if not results:
        print("No results found.")
    else:
        print(retriever.search_json(query))
