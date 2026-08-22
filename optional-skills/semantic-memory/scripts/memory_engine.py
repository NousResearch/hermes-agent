#!/usr/bin/env python3
"""
Hermes Memory Engine (Phase 1-2)
Semantic Retrieval + Temporal Weighting for RAG

Usage:
    python memory_engine.py --init              # Initialize database
    python memory_engine.py --index             # Index all memory files
    python memory_engine.py --query "text"      # Hybrid search
    python memory_engine.py --status            # Show system status
    python memory_engine.py --decay             # Recalculate temporal decay
"""

import sys
import json
import sqlite3
import hashlib
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import argparse
import logging
from dataclasses import dataclass, asdict
import subprocess
import math

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

CONFIG_PATH = Path.home() / ".hermes/memory-engine/config/memory-engine.yaml"
DB_PATH = Path.home() / ".hermes/memory-engine/db/memory.db"
LOG_PATH = Path.home() / ".hermes/memory-engine/logs/memory-engine.log"

# Create logs directory
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────

@dataclass
class MemoryFact:
    """Represents a single memory fact"""
    id: str
    content: str
    source: str
    created_at: str
    updated_at: str
    confidence: float = 0.9
    decay_weight: float = 1.0
    referenced_count: int = 0
    fact_type: str = "UNKNOWN"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class SearchResult:
    """Result of a hybrid search query"""
    fact_id: str
    content: str
    source: str
    bm25_score: float
    semantic_score: float
    temporal_score: float
    combined_score: float
    decay_weight: float
    freshness_tier: str


# ─────────────────────────────────────────────────────────────────
# MEMORY ENGINE CLASS
# ─────────────────────────────────────────────────────────────────

class MemoryEngine:
    """Main memory engine with semantic retrieval and temporal weighting"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.config = self._load_config()
        self.conn = None
        
    def _load_config(self) -> dict:
        """Load configuration from YAML"""
        try:
            with open(CONFIG_PATH, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config not found at {CONFIG_PATH}")
            return {}
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def init_db(self):
        """Initialize database schema"""
        schema_path = Path.home() / ".hermes/memory-engine/db/schema.sql"
        
        try:
            with open(schema_path, 'r') as f:
                schema = f.read()
            
            cursor = self.conn.cursor()
            cursor.executescript(schema)
            self.conn.commit()
            logger.info("✓ Database schema initialized")
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise
    
    def generate_fact_id(self, content: str) -> str:
        """Generate unique fact ID from content hash"""
        return f"fact_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def get_content_hash(self, content: str) -> str:
        """Generate content hash for deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def calculate_decay_weight(self, created_at: datetime) -> float:
        """
        Calculate temporal decay weight using exponential function.
        weight = e^(-lambda * days_old) with lambda=0.05
        
        Recent (< 7 days):  weight ≈ 0.7-1.0
        Medium (7-30 days): weight ≈ 0.3-0.7
        Old (> 30 days):    weight ≈ 0.1-0.3
        """
        days_old = (datetime.now() - created_at).days
        lambda_param = self.config.get('TEMPORAL_WEIGHTING', {}).get('decay_lambda', 0.05)
        decay_weight = math.exp(-lambda_param * days_old)
        return max(0.01, decay_weight)  # never go below 0.01
    
    def get_freshness_tier(self, created_at: datetime) -> str:
        """Classify fact into freshness tier"""
        days_old = (datetime.now() - created_at).days
        
        if days_old <= 7:
            return "recent"
        elif days_old <= 30:
            return "medium"
        elif days_old <= 90:
            return "old"
        else:
            return "archive"
    
    def add_fact(self, 
                 content: str, 
                 source: str, 
                 fact_type: str = "UNKNOWN",
                 tags: List[str] = None,
                 confidence: float = 0.9) -> str:
        """Add a new fact to the engine"""
        if tags is None:
            tags = []
        
        fact_id = self.generate_fact_id(content)
        content_hash = self.get_content_hash(content)
        created_at = datetime.now()
        
        # Check for duplicates
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM memory_facts WHERE content_hash = ?",
            (content_hash,)
        )
        existing = cursor.fetchone()
        
        if existing:
            logger.warning(f"Duplicate fact detected: {fact_id} (existing: {existing[0]})")
            return existing[0]
        
        # Calculate temporal properties
        decay_weight = self.calculate_decay_weight(created_at)
        freshness_tier = self.get_freshness_tier(created_at)
        
        try:
            cursor.execute("""
                INSERT INTO memory_facts (
                    id, content, source, created_at, updated_at,
                    decay_weight, freshness_tier, fact_type, tags,
                    confidence, content_hash, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact_id,
                content,
                source,
                created_at.isoformat(),
                created_at.isoformat(),
                decay_weight,
                freshness_tier,
                fact_type,
                json.dumps(tags),
                confidence,
                content_hash,
                1
            ))
            self.conn.commit()
            logger.info(f"✓ Added fact: {fact_id} (source: {source})")
            return fact_id
        except sqlite3.Error as e:
            logger.error(f"Failed to add fact: {e}")
            raise
    
    def bm25_score(self, query: str, content: str) -> float:
        """
        Simple BM25-like scoring (lexical relevance)
        
        Higher score if:
        - Query terms appear in content
        - Query terms appear multiple times
        - Substring matches count (e.g. "auth" in "authentication")
        """
        query_terms = set(query.lower().split())
        content_lower = content.lower()
        content_words = set(content_lower.split())
        
        if not query_terms:
            return 0.0
        
        matched = 0
        score = 0.0
        for term in query_terms:
            # Exact word match
            if term in content_words:
                score += 1.0
                matched += 1
            # Substring match (e.g. "auth" in "authentication")
            elif term in content_lower:
                score += 0.8
                matched += 1
            # Check if any content word starts with query term
            elif any(w.startswith(term) for w in content_words):
                score += 0.7
                matched += 1
        
        # Normalize by number of query terms
        return score / len(query_terms)
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding using FastEmbed (local, free)."""
        try:
            from embedder import Embedder
            emb = Embedder()
            return emb.embed(text)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None
    
    def get_semantic_score(self, query_vec: List[float], content: str) -> float:
        """Compute semantic similarity between query vector and content."""
        try:
            from embedder import Embedder, EmbeddingStore
            emb = Embedder()
            content_vec = emb.embed(content)
            return emb.cosine_similarity(query_vec, content_vec)
        except Exception:
            return 0.5  # fallback
    
    def hybrid_search(self, query: str, top_k: int = 5, use_embeddings: bool = True) -> List[SearchResult]:
        """
        Hybrid search combining:
        1. BM25 (lexical relevance) - 40%
        2. Semantic similarity (FastEmbed) - 40%
        3. Temporal weighting - 20%
        """
        cursor = self.conn.cursor()
        
        # Pre-compute query embedding once
        query_vec = None
        embedding_cache = {}
        if use_embeddings:
            try:
                from embedder import Embedder, EmbeddingStore
                emb = Embedder()
                query_vec = emb.embed(query)
                
                # Load all stored embeddings in one shot
                store_conn = sqlite3.connect(str(self.db_path))
                rows = store_conn.execute(
                    "SELECT fact_id, embedding FROM fact_embeddings"
                ).fetchall()
                store_conn.close()
                for fid, evec in rows:
                    embedding_cache[fid] = json.loads(evec)
            except Exception as e:
                logger.warning(f"Embeddings unavailable, falling back: {e}")
                query_vec = None
        
        # Get all active facts
        cursor.execute("""
            SELECT id, content, source, created_at, decay_weight, freshness_tier
            FROM memory_facts
            WHERE is_active = 1 AND is_archived = 0
            ORDER BY created_at DESC
        """)
        
        facts = cursor.fetchall()
        results = []
        
        for fact in facts:
            fact_id, content, source, created_at, decay_weight, freshness_tier = fact
            
            # BM25 scoring
            bm25_score = self.bm25_score(query, content)
            
            # Semantic scoring (real embeddings or fallback)
            if query_vec and fact_id in embedding_cache:
                from embedder import Embedder
                semantic_score = Embedder.cosine_similarity(query_vec, embedding_cache[fact_id])
                # Normalize: cosine sim can be negative, clamp to [0, 1]
                semantic_score = max(0.0, semantic_score)
            else:
                semantic_score = 0.3  # low fallback for unembedded facts
            
            # Temporal scoring (freshness bonus)
            created = datetime.fromisoformat(created_at)
            temporal_score = self.calculate_decay_weight(created)
            
            # Weights
            bm25_weight = 0.4
            semantic_weight = 0.4
            temporal_weight = 0.2
            
            # Combined score
            combined_score = (
                bm25_score * bm25_weight +
                semantic_score * semantic_weight +
                temporal_score * temporal_weight
            )
            
            if combined_score > 0.15:  # lower threshold now that semantic is real
                results.append(SearchResult(
                    fact_id=fact_id,
                    content=content,
                    source=source,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                    temporal_score=temporal_score,
                    combined_score=combined_score,
                    decay_weight=decay_weight,
                    freshness_tier=freshness_tier
                ))
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results[:top_k]
    
    def update_decay_weights(self):
        """Recalculate temporal decay weights for all facts"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT id, created_at FROM memory_facts WHERE is_active = 1")
        facts = cursor.fetchall()
        
        updated_count = 0
        for fact_id, created_at in facts:
            created = datetime.fromisoformat(created_at)
            new_weight = self.calculate_decay_weight(created)
            new_tier = self.get_freshness_tier(created)
            
            cursor.execute("""
                UPDATE memory_facts
                SET decay_weight = ?, freshness_tier = ?, updated_at = ?
                WHERE id = ?
            """, (new_weight, new_tier, datetime.now().isoformat(), fact_id))
            
            updated_count += 1
        
        self.conn.commit()
        logger.info(f"✓ Updated decay weights for {updated_count} facts")
    
    def get_status(self) -> dict:
        """Get engine status"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM memory_facts WHERE is_active = 1")
        total_facts = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN freshness_tier = 'recent' THEN 1 ELSE 0 END) as recent,
                SUM(CASE WHEN freshness_tier = 'medium' THEN 1 ELSE 0 END) as medium,
                SUM(CASE WHEN freshness_tier = 'old' THEN 1 ELSE 0 END) as old,
                SUM(CASE WHEN freshness_tier = 'archive' THEN 1 ELSE 0 END) as archive
            FROM memory_facts
            WHERE is_active = 1
        """)
        tiers = dict(cursor.fetchone())
        
        cursor.execute("SELECT COUNT(*) FROM memory_facts WHERE is_archived = 1")
        archived = cursor.fetchone()[0]
        
        return {
            "status": "operational",
            "total_facts": total_facts,
            "freshness_distribution": tiers,
            "archived_facts": archived,
            "database_path": str(self.db_path),
            "timestamp": datetime.now().isoformat()
        }


# ─────────────────────────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hermes Memory Engine (Phase 1-2)"
    )
    parser.add_argument('--init', action='store_true', help='Initialize database')
    parser.add_argument('--index', action='store_true', help='Index memory files')
    parser.add_argument('--query', type=str, help='Hybrid search query')
    parser.add_argument('--status', action='store_true', help='Show engine status')
    parser.add_argument('--decay', action='store_true', help='Recalculate temporal decay')
    parser.add_argument('--add-fact', type=str, help='Add a single fact')
    parser.add_argument('--source', type=str, default='manual', help='Source for fact')
    
    args = parser.parse_args()
    
    engine = MemoryEngine()
    
    try:
        engine.connect()
        
        if args.init:
            engine.init_db()
            print("✓ Database initialized")
        
        elif args.query:
            results = engine.hybrid_search(args.query, top_k=5)
            print(f"\n╔═══════════════════════════════════════════════════════╗")
            print(f"║ HYBRID SEARCH RESULTS: '{args.query}'")
            print(f"╚═══════════════════════════════════════════════════════╝\n")
            
            for i, result in enumerate(results, 1):
                print(f"[{i}] {result.fact_id} | {result.source}")
                print(f"    Score: {result.combined_score:.2%} (BM25: {result.bm25_score:.2%}, Semantic: {result.semantic_score:.2%}, Temporal: {result.temporal_score:.2%})")
                print(f"    Freshness: {result.freshness_tier} | Weight: {result.decay_weight:.3f}")
                print(f"    Content: {result.content[:100]}...")
                print()
        
        elif args.status:
            status = engine.get_status()
            print("\n╔═══════════════════════════════════════════════════════╗")
            print("║ MEMORY ENGINE STATUS")
            print("╚═══════════════════════════════════════════════════════╝\n")
            print(json.dumps(status, indent=2))
        
        elif args.decay:
            engine.update_decay_weights()
            print("✓ Temporal decay weights updated")
        
        elif args.add_fact:
            fact_id = engine.add_fact(
                content=args.add_fact,
                source=args.source,
                fact_type="MANUAL"
            )
            print(f"✓ Fact added: {fact_id}")
        
        else:
            print("Memory Engine Phase 1-2")
            print("Use --help for options")
    
    finally:
        engine.close()


if __name__ == "__main__":
    main()
