#!/usr/bin/env python3
"""
neural_memory.py - THE Unified Neural Memory API
One import to rule them all.

Usage:
    from neural_memory import Memory
    
    mem = Memory()
    mem.remember("The user has a dog named Lou")
    results = mem.recall("What pet does the user have?")
    mem.think(results[0].id)
    mem.consolidate()
    mem.close()
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add python dir to path
sys.path.insert(0, str(Path(__file__).parent))

from embed_provider import EmbeddingProvider
from memory_client import NeuralMemory, SQLiteStore

# Try MSSQL, fall back to SQLite
try:
    from mssql_store import MSSQLStore
    HAS_MSSQL = True
except ImportError:
    HAS_MSSQL = False


class Memory:
    """
    Unified Neural Memory interface.
    
    Auto-detects best available backend:
    1. C++ library + sentence-transformers (fastest)
    2. Pure Python + sentence-transformers (good)
    3. Pure Python + TF-IDF (works everywhere)
    4. Pure Python + hash (zero dependencies)
    """
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 embedding_backend: str = "auto",
                 use_cpp: bool = True,
                 use_mssql: bool = False,
                 mssql_server: str = "localhost",
                 mssql_database: str = "NeuralMemory"):
        
        Path.home().joinpath(".neural_memory").mkdir(parents=True, exist_ok=True)
        
        self._db_path = db_path or str(Path.home() / ".neural_memory" / "memory.db")
        self._embedding_backend = embedding_backend
        self._cpp = None
        self._python = None
        self._mssql = None
        
        # MSSQL or SQLite?
        self._use_mssql = use_mssql and HAS_MSSQL
        
        # Try C++ backend first
        if use_cpp:
            try:
                from cpp_bridge import NeuralMemoryCpp
                self._cpp = NeuralMemoryCpp()
                self._cpp.initialize(dim=384)
                print(f"[neural] C++ backend loaded")
            except (FileNotFoundError, OSError, ImportError) as e:
                print(f"[neural] C++ not available: {e}")
                self._cpp = None
        
        # Fallback to Python backend
        if self._cpp is None:
            self._python = NeuralMemory(db_path=self._db_path, embedding_backend=embedding_backend)
            print(f"[neural] Python backend: {self._python.embedder.backend.__class__.__name__}")
        
        # MSSQL store (if requested)
        if self._use_mssql:
            try:
                self._mssql = MSSQLStore(
                    server=mssql_server,
                    database=mssql_database
                )
                print(f"[neural] MSSQL: {mssql_server}/{mssql_database}")
            except Exception as e:
                print(f"[neural] MSSQL failed: {e}, using SQLite")
                self._mssql = None
        
        # Always use embed_provider for text->vector
        if self._python:
            self._embedder = self._python.embedder
        else:
            self._embedder = EmbeddingProvider(backend=embedding_backend)
        
        self._dim = self._embedder.dim
    
    def remember(self, text: str, label: str = "") -> int:
        """
        Store a memory from text.
        Returns memory ID.
        """
        embedding = self._embedder.embed(text)
        
        if self._cpp:
            mid = self._cpp.store(embedding, label or text[:60], text)
        else:
            mid = self._python.remember(text, label)
        
        # Also store in MSSQL if available
        if self._mssql:
            try:
                self._mssql.store(label or text[:60], text, embedding)
            except Exception:
                pass
        
        return mid
    
    def remember_embedding(self, embedding: list[float], label: str = "", 
                           content: str = "") -> int:
        """Store a memory with pre-computed embedding."""
        if self._cpp:
            return self._cpp.store(embedding, label, content)
        else:
            mid = self._python.store.store(label or content[:60], content, embedding)
            # Update in-memory graph
            self._python._graph_nodes[mid] = {
                'embedding': embedding,
                'label': label or content[:60],
                'connections': {}
            }
            # Auto-connect
            for other_id, other_node in self._python._graph_nodes.items():
                if other_id == mid:
                    continue
                sim = NeuralMemory._cosine_similarity(embedding, other_node['embedding'])
                if sim > 0.15:
                    self._python._graph_nodes[mid]['connections'][other_id] = sim
                    self._python._graph_nodes[other_id]['connections'][mid] = sim
                    self._python.store.add_connection(mid, other_id, sim)
            return mid
    
    def recall(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrieve memories related to query text.
        Returns list of {id, label, content, similarity, connections}.
        """
        embedding = self._embedder.embed(query)
        
        if self._cpp:
            raw = self._cpp.retrieve(embedding, k)
            return [
                {
                    'id': r['id'],
                    'label': r['label'],
                    'content': r['content'],
                    'similarity': r['score'],
                    'connections': [],
                }
                for r in raw
            ]
        else:
            return self._python.recall(query, k)
    
    def think(self, start_id: int, depth: int = 3, decay: float = 0.85) -> list[dict]:
        """
        Spreading activation from a starting memory.
        Returns activated memories sorted by activation.
        """
        if self._cpp:
            raw = self._cpp.think(start_id, depth)
            return [
                {'id': r['id'], 'label': r['label'], 'activation': r['score']}
                for r in raw
            ]
        else:
            return self._python.think(start_id, depth, decay)
    
    def connections(self, mem_id: int) -> list[dict]:
        """Get connections for a memory."""
        if self._python:
            return self._python.connections(mem_id)
        return []
    
    def graph(self) -> dict:
        """Get knowledge graph stats."""
        if self._cpp:
            stats = self._cpp.get_stats()
            return {
                'nodes': stats['graph_nodes'],
                'edges': stats['graph_edges'],
                'hopfield_patterns': stats['hopfield_patterns'],
            }
        else:
            return self._python.graph()
    
    def consolidate(self) -> int:
        """Run memory consolidation."""
        if self._cpp:
            return self._cpp.consolidate()
        # Python mode: nothing to consolidate (SQLite is already persistent)
        return 0
    
    def stats(self) -> dict:
        """Get system statistics."""
        if self._cpp:
            return self._cpp.get_stats()
        else:
            s = self._python.stats()
            s['backend'] = 'python'
            return s
    
    def close(self):
        """Clean shutdown."""
        if self._cpp:
            self._cpp.shutdown()
            self._cpp = None
        if self._mssql:
            self._mssql.close()
            self._mssql = None
        if self._python:
            self._python.close()
            self._python = None
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def backend(self) -> str:
        if self._cpp:
            return "cpp"
        return self._embedder.backend.__class__.__name__
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __repr__(self):
        return f"Memory(backend={self.backend}, dim={self.dim}, db={self._db_path})"
