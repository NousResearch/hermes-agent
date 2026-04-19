#!/usr/bin/env python3
"""
memory_client.py - Python client for Neural Memory Adapter
Wraps the C++ library via ctypes. Uses embed_provider for text->vector.
"""

import ctypes
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Optional

# ============================================================================
# Find the shared library
# ============================================================================

def _find_lib():
    candidates = [
        Path(__file__).parent.parent / "build" / "libneural_memory.so",
        Path.home() / "projects" / "neural-memory-adapter" / "build" / "libneural_memory.so",
        Path("/usr/local/lib/libneural_memory.so"),
        Path("/usr/lib/libneural_memory.so"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("libneural_memory.so not found. Build first: cd build && cmake --build .")

# ============================================================================
# C API struct definitions
# ============================================================================

class CSearchResult(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("score", ctypes.c_float),
        ("label", ctypes.c_char * 256),
        ("content", ctypes.c_char * 1024),
    ]

class CStats(ctypes.Structure):
    _fields_ = [
        ("total_stores", ctypes.c_uint64),
        ("total_retrieves", ctypes.c_uint64),
        ("total_searches", ctypes.c_uint64),
        ("total_consolidations", ctypes.c_uint64),
        ("avg_store_us", ctypes.c_uint64),
        ("avg_retrieve_us", ctypes.c_uint64),
        ("graph_nodes", ctypes.c_size_t),
        ("graph_edges", ctypes.c_size_t),
        ("hopfield_patterns", ctypes.c_size_t),
        ("hopfield_occupancy", ctypes.c_float),
    ]

# ============================================================================
# SQLite persistence layer
# ============================================================================

DB_PATH = Path.home() / ".neural_memory" / "memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    content TEXT,
    embedding BLOB,
    salience REAL DEFAULT 1.0,
    created_at REAL DEFAULT (unixepoch()),
    last_accessed REAL DEFAULT (unixepoch()),
    access_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS connections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER,
    target_id INTEGER,
    weight REAL DEFAULT 0.5,
    edge_type TEXT DEFAULT 'similar',
    created_at REAL DEFAULT (unixepoch()),
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_connections_source ON connections(source_id);
CREATE INDEX IF NOT EXISTS idx_connections_target ON connections(target_id);
"""

class SQLiteStore:
    def __init__(self, db_path: str | Path = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read perf
        self.conn.execute("PRAGMA synchronous=NORMAL") # Faster writes
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._lock = threading.Lock()
    
    def store(self, label: str, content: str, embedding: list[float]) -> int:
        blob = struct.pack(f'{len(embedding)}f', *embedding)
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO memories (label, content, embedding) VALUES (?, ?, ?)",
                (label, content, blob)
            )
            self.conn.commit()
            return cur.lastrowid
    
    def get_all(self) -> list[dict]:
        import struct
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, label, content, embedding, salience, access_count FROM memories ORDER BY id"
            ).fetchall()
        results = []
        for row in rows:
            id_, label, content, blob, salience, access_count = row
            if blob is None:
                continue
            dim = len(blob) // 4
            embedding = list(struct.unpack(f'{dim}f', blob))
            results.append({
                'id': id_, 'label': label, 'content': content,
                'embedding': embedding, 'salience': salience,
                'access_count': access_count
            })
        return results
    
    def get(self, id_: int) -> Optional[dict]:
        import struct
        row = self.conn.execute(
            "SELECT id, label, content, embedding, salience, access_count FROM memories WHERE id = ?",
            (id_,)
        ).fetchone()
        if not row:
            return None
        id_, label, content, blob, salience, access_count = row
        dim = len(blob) // 4
        embedding = list(struct.unpack(f'{dim}f', blob))
        return {
            'id': id_, 'label': label, 'content': content,
            'embedding': embedding, 'salience': salience,
            'access_count': access_count
        }
    
    def touch(self, id_: int):
        with self._lock:
            self.conn.execute(
                "UPDATE memories SET last_accessed = unixepoch(), access_count = access_count + 1 WHERE id = ?",
                (id_,)
            )
            self.conn.commit()
    
    def add_connection(self, source: int, target: int, weight: float, edge_type: str = "similar"):
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO connections (source_id, target_id, weight, edge_type) VALUES (?, ?, ?, ?)",
                (source, target, weight, edge_type)
            )
            self.conn.commit()
    
    def get_connections(self, node_id: int) -> list[dict]:
        rows = self.conn.execute(
            """SELECT source_id, target_id, weight, edge_type FROM connections 
               WHERE source_id = ? OR target_id = ? ORDER BY weight DESC""",
            (node_id, node_id)
        ).fetchall()
        return [
            {'source': r[0], 'target': r[1], 'weight': r[2], 'type': r[3]}
            for r in rows
        ]
    
    def get_stats(self) -> dict:
        mem_count = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn_count = self.conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
        return {'memories': mem_count, 'connections': conn_count}
    
    def close(self):
        self.conn.close()


# ============================================================================
# Neural Memory Client
# ============================================================================

class NeuralMemory:
    """
    Python interface to the Neural Memory system.
    
    Usage:
        mem = NeuralMemory()
        mem.remember("The user has a dog named Lou")
        mem.remember("Working on BTQuant trading platform")
        results = mem.recall("What pet does the user have?")
    """
    
    def __init__(self, db_path: str | Path = DB_PATH, embedding_backend: str = "auto",
                 use_mssql: bool = False, use_cpp: bool = True):
        from embed_provider import EmbeddingProvider

        self.embedder = EmbeddingProvider(backend=embedding_backend)

        if use_mssql:
            from mssql_store import MSSQLStore
            self.store = MSSQLStore()
        else:
            self.store = SQLiteStore(db_path)

        self.dim = self.embedder.dim

        # C++ SIMD index for fast retrieval (primary search path)
        self._cpp = None
        if use_cpp:
            try:
                from cpp_bridge import NeuralMemoryCpp
                self._cpp = NeuralMemoryCpp()
                self._cpp.initialize(dim=self.dim)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "C++ bridge unavailable, falling back to Python: %s", e
                )
                self._cpp = None

        # In-memory graph for spreading activation
        self._graph_nodes: dict[int, dict] = {}  # id -> {embedding, connections}
        self._load_from_store()

    def _load_from_store(self):
        """Load existing memories into in-memory graph + C++ index."""
        all_mems = self.store.get_all()
        for mem in all_mems:
            self._graph_nodes[mem['id']] = {
                'embedding': mem['embedding'],
                'label': mem['label'],
                'connections': {}
            }

        # Load connections
        for mem in all_mems:
            conns = self.store.get_connections(mem['id'])
            for c in conns:
                other = c['target'] if c['source'] == mem['id'] else c['source']
                self._graph_nodes[mem['id']]['connections'][other] = c['weight']

        # Load into C++ SIMD index + build ID mapping
        self._cpp_id_map = {}  # cpp_id -> sqlite_id
        if self._cpp:
            for mem in all_mems:
                emb = mem.get('embedding', [])
                if emb and len(emb) == self.dim:
                    cpp_id = self._cpp.store(emb, mem.get('label', ''), mem.get('content', ''))
                    self._cpp_id_map[cpp_id] = mem['id']
    
    def remember(self, text: str, label: str = "", detect_conflicts: bool = True) -> int:
        """Store a memory. Returns memory ID.
        
        If detect_conflicts=True, checks for existing memories about the same
        topic that contain contradictory information and updates/invalidates them.
        """
        import math
        import time
        
        embedding = self.embedder.embed(text)
        
        # Knowledge-update: detect conflicts with existing memories
        if detect_conflicts and self._graph_nodes:
            conflicts = self._find_conflicts(text, embedding)
            for conflict_id, conflict_info in conflicts.items():
                old_content = conflict_info['content']
                similarity = conflict_info['similarity']
                
                # High similarity + different content = likely update
                if similarity > 0.7 and self._content_differs(old_content, text):
                    # Mark old memory as superseded
                    with self.store._lock:
                        self.store.conn.execute(
                            "UPDATE memories SET content = ? WHERE id = ?",
                            (f"[SUPERSEDED] {old_content}\n[UPDATED TO] {text}", conflict_id)
                        )
                        self.store.conn.commit()
                    # Update in-memory graph
                    if conflict_id in self._graph_nodes:
                        self._graph_nodes[conflict_id]['embedding'] = embedding
                    # Don't create duplicate - update existing
                    return conflict_id
        
        mem_id = self.store.store(label or text[:60], text, embedding)
        
        # Add to in-memory graph
        self._graph_nodes[mem_id] = {
            'embedding': embedding,
            'label': label or text[:60],
            'connections': {}
        }

        # Add to C++ SIMD index + track mapping
        if self._cpp:
            try:
                cpp_id = self._cpp.store(embedding, label or text[:60], text)
                self._cpp_id_map[cpp_id] = mem_id
            except Exception:
                pass

        # Auto-connect to similar memories
        for other_id, other_node in self._graph_nodes.items():
            if other_id == mem_id:
                continue
            sim = self._cosine_similarity(embedding, other_node['embedding'])
            if sim > 0.15:  # Threshold for connection
                self._graph_nodes[mem_id]['connections'][other_id] = sim
                self._graph_nodes[other_id]['connections'][mem_id] = sim
                self.store.add_connection(mem_id, other_id, sim)

        return mem_id
    
    def _find_conflicts(self, new_text: str, new_embedding: list[float], threshold: float = 0.6) -> dict:
        """Find memories that might conflict with the new text.
        Returns {memory_id: {similarity, content}} for potential conflicts.
        """
        conflicts = {}
        for mem in self.store.get_all():
            sim = self._cosine_similarity(new_embedding, mem['embedding'])
            if sim > threshold:
                conflicts[mem['id']] = {
                    'similarity': sim,
                    'content': mem['content'],
                    'label': mem['label']
                }
        return conflicts
    
    def _content_differs(self, old_text: str, new_text: str) -> bool:
        """Check if two texts contain different information despite high similarity.
        Heuristics: different numbers, dates, negations, or significant word differences.
        """
        import re
        
        old_clean = old_text.replace("[SUPERSEDED]", "").replace("[UPDATED TO]", "").strip()
        
        # Extract numbers from both
        old_nums = set(re.findall(r'\d+\.?\d*', old_clean))
        new_nums = set(re.findall(r'\d+\.?\d*', new_text))
        
        # Different numbers = different info
        if old_nums != new_nums and old_nums and new_nums:
            return True
        
        # Check for negation differences
        negations = {'not', "n't", 'never', 'no', 'none', 'nothing', 'nowhere'}
        old_neg = any(n in old_clean.lower().split() for n in negations)
        new_neg = any(n in new_text.lower().split() for n in negations)
        if old_neg != new_neg:
            return True
        
        # Check for date differences
        old_dates = set(re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}', old_clean))
        new_dates = set(re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}', new_text))
        if old_dates != new_dates and old_dates and new_dates:
            return True
        
        # Check for significant content word differences (excluding common words)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                     'on', 'with', 'at', 'by', 'from', 'it', 'its', "it's", 'this', 'that',
                     'user', 'user\'s', 'my', 'i', 'me', 'we', 'our', 'you', 'your'}
        
        def extract_keywords(text):
            words = set(re.findall(r'\b[a-z]+\b', text.lower()))
            return words - stopwords
        
        old_kw = extract_keywords(old_clean)
        new_kw = extract_keywords(new_text)
        
        # If more than 30% of keywords differ, it's a real update
        if old_kw and new_kw:
            shared = old_kw & new_kw
            total = old_kw | new_kw
            diff_ratio = 1.0 - len(shared) / len(total)
            if diff_ratio > 0.3:
                return True
        
        return False
    
    def recall(self, query: str, k: int = 5, temporal_weight: float = 0.2) -> list[dict]:
        """
        Retrieve memories related to query.

        Args:
            query: Search query
            k: Number of results
            temporal_weight: Weight for recency scoring (0=pure similarity, 1=pure recency)

        Returns list of {id, label, content, similarity, temporal_score, connections}.
        """
        import math
        import time

        query_vec = self.embedder.embed(query)
        now = time.time()

        # C++ fast path: SIMD retrieve returns top-k candidates in microseconds
        # Then apply temporal scoring on the small candidate set
        if self._cpp:
            try:
                candidates = self._cpp.retrieve(query_vec, k=k * 3)
                if candidates:
                    scored = []
                    for c in candidates:
                        cpp_id = c['id']
                        # Map C++ index back to SQLite ID
                        mem_id = self._cpp_id_map.get(cpp_id, cpp_id)
                        sim = c.get('similarity', c.get('score', 0))
                        node = self._graph_nodes.get(mem_id, {})

                        scored.append({
                            'id': mem_id,
                            'label': c.get('label', node.get('label', '')),
                            'content': c.get('content', ''),
                            'embedding': node.get('embedding', []),
                            'similarity': sim,
                            'temporal_score': 0.5,
                            'combined': (1 - temporal_weight) * sim + temporal_weight * 0.5,
                            'connections': list(node.get('connections', {}).keys()),
                        })

                    scored.sort(key=lambda x: -x['combined'])
                    # Touch accessed memories
                    for s in scored[:k]:
                        try:
                            self.store.touch(s['id'])
                        except Exception:
                            pass
                    return scored[:k]
            except Exception:
                pass  # Fall through to Python path

        # Python fallback: O(n) linear scan
        scored = []
        for mem in self.store.get_all():
            sim = self._cosine_similarity(query_vec, mem['embedding'])

            # Temporal score: exponential decay based on last_accessed
            try:
                row = self.store.conn.execute(
                    "SELECT last_accessed FROM memories WHERE id = ?", (mem['id'],)
                ).fetchone()
                if row and row[0]:
                    age_hours = (now - row[0]) / 3600
                    temporal_score = math.exp(-0.693 * age_hours / 24)
                else:
                    temporal_score = 0.5
            except Exception:
                temporal_score = 0.5

            combined = (1 - temporal_weight) * sim + temporal_weight * temporal_score
            scored.append({**mem, 'similarity': sim, 'temporal_score': temporal_score, 'combined': combined})

        # Sort by combined score
        scored.sort(key=lambda x: -x['combined'])
        
        # Enrich with connections via spreading activation
        results = []
        seen = set()
        for mem in scored[:k]:
            if mem['id'] in seen:
                continue
            seen.add(mem['id'])
            
            # Get connections
            conns = self.store.get_connections(mem['id'])
            connected = []
            for c in conns:
                other = c['target'] if c['source'] == mem['id'] else c['source']
                if other not in seen:
                    other_mem = self.store.get(other)
                    if other_mem:
                        connected.append({
                            'id': other,
                            'label': other_mem['label'],
                            'weight': c['weight']
                        })
            
            results.append({
                'id': mem['id'],
                'label': mem['label'],
                'content': mem['content'],
                'similarity': round(mem['similarity'], 4),
                'temporal_score': round(mem['temporal_score'], 4),
                'combined': round(mem['combined'], 4),
                'connections': connected[:3],  # Top 3
            })
            
            self.store.touch(mem['id'])
        
        return results
    
    def think(self, start_id: int, depth: int = 3, decay: float = 0.85) -> list[dict]:
        """
        Spreading activation from a starting memory.
        Returns activated memories sorted by activation level.
        """
        if start_id not in self._graph_nodes:
            return []
        
        # BFS with activation propagation
        activation = {start_id: 1.0}
        visited = {start_id}
        queue = [(start_id, 1.0, 0)]
        
        while queue:
            current, act, level = queue.pop(0)
            if level >= depth or act < 0.01:
                continue
            
            node = self._graph_nodes.get(current, {})
            for neighbor_id, weight in node.get('connections', {}).items():
                propagated = act * weight * decay
                if propagated < 0.01:
                    continue
                
                if neighbor_id not in activation or propagated > activation[neighbor_id]:
                    activation[neighbor_id] = propagated
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, propagated, level + 1))
        
        # Build results (skip start node)
        results = []
        for node_id, act in activation.items():
            if node_id == start_id:
                continue
            mem = self.store.get(node_id)
            if mem:
                results.append({
                    'id': node_id,
                    'label': mem['label'],
                    'activation': round(act, 4),
                })
        
        results.sort(key=lambda x: -x['activation'])
        return results
    
    def connections(self, mem_id: int) -> list[dict]:
        """Get all connections for a memory."""
        conns = self.store.get_connections(mem_id)
        results = []
        for c in conns:
            other = c['target'] if c['source'] == mem_id else c['source']
            mem = self.store.get(other)
            if mem:
                results.append({
                    'id': other,
                    'label': mem['label'],
                    'weight': round(c['weight'], 4),
                    'type': c['type']
                })
        return results
    
    def graph(self) -> dict:
        """Get knowledge graph stats and structure."""
        stats = self.store.get_stats()
        
        # Build adjacency summary
        edges = []
        seen = set()
        for node_id, node in self._graph_nodes.items():
            for other_id, weight in node.get('connections', {}).items():
                key = tuple(sorted([node_id, other_id]))
                if key not in seen:
                    seen.add(key)
                    edges.append({
                        'from': node_id,
                        'to': other_id,
                        'weight': round(weight, 3)
                    })
        
        return {
            'nodes': stats['memories'],
            'edges': len(edges),
            'top_edges': sorted(edges, key=lambda x: -x['weight'])[:10]
        }
    
    def stats(self) -> dict:
        """Get memory statistics."""
        graph = self.store.stats() if hasattr(self.store, 'stats') else self.store.get_stats()
        return {
            'memories': graph['memories'],
            'connections': graph['connections'],
            'embedding_dim': self.dim,
            'embedding_backend': self.embedder.backend.__class__.__name__,
        }
    
    def close(self):
        if self._cpp:
            try:
                self._cpp.shutdown()
            except Exception:
                pass
            self._cpp = None
        self.store.close()
    
    # Cython-accelerated ops (falls back to Python if unavailable)
    try:
        from fast_ops import cosine_similarity as _cosine_sim_fast
    except ImportError:
        _cosine_sim_fast = None

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        if NeuralMemory._cosine_sim_fast is not None:
            import numpy as np
            # Avoid repeated array creation for lists
            if not isinstance(a, np.ndarray):
                a = np.asarray(a, dtype=np.float64)
            if not isinstance(b, np.ndarray):
                b = np.asarray(b, dtype=np.float64)
            return float(NeuralMemory._cosine_sim_fast(a, b))
        dot = sum(x*y for x, y in zip(a, b))
        na = (sum(x*x for x in a)) ** 0.5
        nb = (sum(x*x for x in b)) ** 0.5
        return dot / (na * nb) if na and nb else 0.0
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
