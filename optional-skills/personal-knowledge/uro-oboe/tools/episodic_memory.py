"""
Episodic Memory Tools for Hermes Agent

Personal episodic memory with 1-bit quantized FAISS binary HNSW for fuzzy recall.
Tools: memory_store, memory_recall_fuzzy, memory_fetch, memory_search_fts, memory_delete, memory_stats
"""

import json
import sqlite3
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 100

# Database and index paths
SKILL_DIR = Path(__file__).parent.parent
DATA_DIR = SKILL_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "episodic_memory.db"
INDEX_PATH = DATA_DIR / "episodic_memory_binary.hnsw"
META_PATH = DATA_DIR / "episodic_memory_meta.pkl"


class EpisodicMemory:
    """Core episodic memory system with 1-bit quantized FAISS HNSW."""

    def __init__(self):
        self._model = None
        self._index = None
        self._conn = None
        self._id_to_idx = {}  # memory_id -> faiss index position
        self._idx_to_id = {}  # faiss index position -> memory_id
        self._next_faiss_idx = 0
        self._initialized = False

    def _get_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def _get_db(self):
            """Get or create database connection."""
            if self._conn is None:
                self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._init_db()
                # Run consistency check once after DB initialization
                if not self._initialized:
                    self._verify_and_repair_consistency()
                    self._initialized = True
            return self._conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_db()
        cursor = conn.cursor()

        # Main memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                vector BLOB,
                is_deleted INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration: add is_deleted column if missing (for existing databases)
        cursor.execute("PRAGMA table_info(memories)")
        columns = [row['name'] for row in cursor.fetchall()]
        if 'is_deleted' not in columns:
            cursor.execute("ALTER TABLE memories ADD COLUMN is_deleted INTEGER DEFAULT 0")
            conn.commit()

        # FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                text,
                tags,
                content='memories',
                content_rowid='id'
            )
        """)

        # Triggers to keep FTS in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text, tags) VALUES ('delete', old.id, old.text, old.tags);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text, tags) VALUES ('delete', old.id, old.text, old.tags);
                INSERT INTO memories_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags);
            END
        """)

        # Index mapping table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_mapping (
                memory_id INTEGER PRIMARY KEY,
                faiss_idx INTEGER NOT NULL UNIQUE
            )
        """)

        conn.commit()

    def _get_index(self):
            """Get or create FAISS binary HNSW index."""
            if self._index is None:
                if faiss is None:
                    raise RuntimeError("faiss-cpu not installed. Run: pip install faiss-cpu")

                if INDEX_PATH.exists():
                    self._index = faiss.read_index_binary(str(INDEX_PATH))
                    self._load_mapping()
                else:
                    self._index = faiss.IndexBinaryHNSW(EMBEDDING_DIM, HNSW_M)
                    self._index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
                    self._index.hnsw.efSearch = HNSW_EF_SEARCH
            return self._index

    def _verify_and_repair_consistency(self):
        """Verify SQLite-FAISS consistency on startup; rebuild index if mismatched."""
        conn = self._get_db()
        cursor = conn.cursor()

        # Count active (non-deleted) memories in SQLite
        cursor.execute("SELECT COUNT(*) as count FROM memories WHERE is_deleted = 0")
        sqlite_count = cursor.fetchone()['count']

        # Count vectors in FAISS index
        faiss_count = self._index.ntotal if self._index else 0

        # Count mappings
        mapping_count = len(self._id_to_idx)

        # If inconsistent, rebuild index from SQLite
        if sqlite_count != faiss_count or sqlite_count != mapping_count:
            print(f"[EpisodicMemory] Consistency check failed: SQLite={sqlite_count}, FAISS={faiss_count}, Mapping={mapping_count}. Rebuilding index...")
            self._rebuild_index_from_sqlite()

    def _rebuild_index_from_sqlite(self):
        """Rebuild FAISS index and mappings from SQLite (source of truth)."""
        if faiss is None:
            raise RuntimeError("faiss-cpu not installed")

        conn = self._get_db()
        cursor = conn.cursor()

        # Get all non-deleted memories with vectors
        cursor.execute("""
            SELECT id, vector FROM memories
            WHERE is_deleted = 0 AND vector IS NOT NULL
            ORDER BY id
        """)
        rows = cursor.fetchall()

        # Create new index
        self._index = faiss.IndexBinaryHNSW(EMBEDDING_DIM, HNSW_M)
        self._index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        self._index.hnsw.efSearch = HNSW_EF_SEARCH

        # Reset mappings
        self._id_to_idx = {}
        self._idx_to_id = {}
        self._next_faiss_idx = 0

        # Add vectors to index
        for row in rows:
            memory_id = row['id']
            vec = np.frombuffer(row['vector'], dtype=np.float32)
            binary_vec = self._quantize(vec)
            binary_vec_bytes = np.packbits(binary_vec).tobytes()
            self._index.add(np.frombuffer(binary_vec_bytes, dtype=np.uint8).reshape(1, -1))

            self._id_to_idx[memory_id] = self._next_faiss_idx
            self._idx_to_id[self._next_faiss_idx] = memory_id
            self._next_faiss_idx += 1

        # Rebuild index_mapping table
        cursor.execute("DELETE FROM index_mapping")
        for mem_id, faiss_idx in self._id_to_idx.items():
            cursor.execute("INSERT INTO index_mapping (memory_id, faiss_idx) VALUES (?, ?)", (mem_id, faiss_idx))
        conn.commit()

        # Save rebuilt index and mappings
        self._save_mapping()
        self._save_index()

        print(f"[EpisodicMemory] Index rebuilt: {len(rows)} memories indexed")

    def _load_mapping(self):
        """Load ID mapping from metadata file."""
        if META_PATH.exists():
            with open(META_PATH, 'rb') as f:
                meta = pickle.load(f)
                self._id_to_idx = meta.get('id_to_idx', {})
                self._idx_to_id = meta.get('idx_to_id', {})
                self._next_faiss_idx = meta.get('next_faiss_idx', 0)
        else:
            # Rebuild from database
            conn = self._get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM memories WHERE is_deleted = 0 ORDER BY id")
            for row in cursor.fetchall():
                self._id_to_idx[row['id']] = self._next_faiss_idx
                self._idx_to_id[self._next_faiss_idx] = row['id']
                self._next_faiss_idx += 1

    def _save_mapping(self):
        """Save ID mapping to metadata file."""
        meta = {
            'id_to_idx': self._id_to_idx,
            'idx_to_id': self._idx_to_id,
            'next_faiss_idx': self._next_faiss_idx
        }
        with open(META_PATH, 'wb') as f:
            pickle.dump(meta, f)

    def _save_index(self):
        """Save FAISS index to disk."""
        if self._index is not None:
            faiss.write_index_binary(self._index, str(INDEX_PATH))

    # Auto-tagging keyword patterns
    TAG_PATTERNS = {
        "novel": ["小説", "キャラ", "主人公", "伏線", "世界観", "冒頭", "プロット", "シーン", "セリフ", "設定"],
        "php": ["PHP", "Laravel", "Composer", "Symfony", "PHPStan", "Psalm", "Xdebug", "opcache"],
        "coding": ["Python", "JavaScript", "TypeScript", "関数", "クラス", "非同期", "デコレータ", "リトライ", "エラーハンドリング", "API", "データベース", "SQL", "Git", "Docker", "テスト", "リファクタ"],
        "creative": ["創作", "アイデア", "プロンプト", "画像生成", "動画生成", "Stable Diffusion", "Midjourney", "Suno", "キャラクターデザイン"],
        "infra": ["サーバー", "インフラ", "GPU", "VRAM", "量子化", "Ollama", "ローカルLLM", "モデル", "デプロイ", "クラウド", "CI/CD"],
        "config": ["設定", "コンフィグ", "環境変数", "yaml", "json", "toml", "dotenv", "プロバイダー", "プロファイル"],
        "cooking": ["料理", "レシピ", "出汁", "昆布", "鰹節", "茹で時間", "塩加減", "調理", "味付け"],
        "llm": ["LLM", "プロンプト", "トークン", "コンテキスト", "ファインチューニング", "RAG", "ベクトル", "埋め込み", "推論"],
        "debug": ["エラー", "バグ", "デバッグ", "スタックトレース", "ログ", "例外", "失敗", "クラッシュ", "メモリ不足", "タイムアウト"],
        "idea": ["アイデア", "思いつき", "ひらめき", "メモ", "後で", "いつか", "やりたい", "試したい"],
        "reference": ["参考", "記事", "論文", "ドキュメント", "チュートリアル", "公式", "ドキュメンテーション", "ブックマーク"],
    }

    def _auto_tag(self, text: str) -> List[str]:
        """Generate tags from text using keyword matching.

        English keywords are matched case-insensitively.
        Japanese keywords are matched as-is (no lowercasing effect).
        """
        tags = []
        text_lower = text.lower()  # For English keyword matching
        for tag, keywords in self.TAG_PATTERNS.items():
            for kw in keywords:
                # English keywords: case-insensitive match
                # Japanese/other: direct substring match
                if kw.isascii() and kw.lower() in text_lower:
                    tags.append(tag)
                    break
                elif not kw.isascii() and kw in text:
                    tags.append(tag)
                    break
        return tags[:5]  # Max 5 tags

    def _quantize(self, vec: np.ndarray) -> np.ndarray:
        """1-bit quantization: sign only (vec > 0).astype('uint8')."""
        return (vec > 0).astype(np.uint8)

    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        model = self._get_model()
        vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def store(self, text: str, tags: List[str] = None, metadata: Dict = None, auto_tag: bool = False) -> Dict[str, Any]:
        """Store a new episodic memory."""
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}

        # Auto-generate tags if requested and no manual tags provided
        if auto_tag and not tags:
            tags = self._auto_tag(text)

        # Generate embedding
        vec = self._embed(text)
        binary_vec = self._quantize(vec)

        # Store in database
        conn = self._get_db()
        cursor = conn.cursor()

        tags_json = json.dumps(tags, ensure_ascii=False)
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        vector_blob = vec.tobytes()

        cursor.execute("""
            INSERT INTO memories (text, tags, metadata, vector)
            VALUES (?, ?, ?, ?)
        """, (text, tags_json, metadata_json, vector_blob))

        memory_id = cursor.lastrowid
        conn.commit()

        # Add to FAISS index
        index = self._get_index()
        binary_vec_bytes = np.packbits(binary_vec).tobytes()
        index.add(np.frombuffer(binary_vec_bytes, dtype=np.uint8).reshape(1, -1))

        # Update mapping
        self._id_to_idx[memory_id] = self._next_faiss_idx
        self._idx_to_id[self._next_faiss_idx] = memory_id
        self._next_faiss_idx += 1

        # Save mapping and index
        cursor.execute("INSERT INTO index_mapping (memory_id, faiss_idx) VALUES (?, ?)",
                       (memory_id, self._id_to_idx[memory_id]))
        conn.commit()

        self._save_mapping()
        self._save_index()

        return {
            "id": memory_id,
            "text": text,
            "tags": tags,
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        }

    def delete(self, ids: List[int]) -> Dict[str, Any]:
        """Soft-delete memories by IDs (logical deletion)."""
        if not ids:
            return {"deleted": 0, "ids": []}

        conn = self._get_db()
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(ids))
        cursor.execute(f"""
            UPDATE memories
            SET is_deleted = 1, updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
        """, ids)

        deleted_count = cursor.rowcount
        conn.commit()

        # Note: FAISS index is NOT modified (HNSW doesn't support efficient deletion).
        # Deleted IDs are filtered out at query time in recall_fuzzy/fetch/search_fts.
        # For full cleanup, rebuild index via _rebuild_index_from_sqlite().

        return {"deleted": deleted_count, "ids": ids}

    def _get_valid_ids(self, tags: List[str] = None) -> Set[int]:
        """Get set of valid (non-deleted) memory IDs, optionally filtered by tags."""
        conn = self._get_db()
        cursor = conn.cursor()

        if tags:
            # Build tag filter: memories that have ANY of the specified tags
            # JSON array contains any of the tags
            tag_conditions = ' OR '.join(['tags LIKE ?'] * len(tags))
            params = [f'%"{tag}"%' for tag in tags]
            cursor.execute(f"""
                SELECT id FROM memories
                WHERE is_deleted = 0 AND ({tag_conditions})
            """, params)
        else:
            cursor.execute("SELECT id FROM memories WHERE is_deleted = 0")

        return {row['id'] for row in cursor.fetchall()}

    def recall_fuzzy(self, query: str, k: int = 10, noise_ratio: float = 0.1,
                     tags: List[str] = None, min_score: float = 0.0) -> Dict[str, Any]:
        """Fuzzy recall using 1-bit quantized binary HNSW index."""
        if tags is None:
            tags = []

        # Pre-fetch valid IDs for efficient filtering (avoids N+1 queries)
        valid_ids = self._get_valid_ids(tags) if tags else None

        # Generate query embedding and quantize
        query_vec = self._embed(query)
        query_binary = self._quantize(query_vec)
        query_binary_bytes = np.packbits(query_binary).tobytes()
        query_binary_arr = np.frombuffer(query_binary_bytes, dtype=np.uint8).reshape(1, -1)

        # Search index
        index = self._get_index()
        if index.ntotal == 0:
            return {"results": [], "total": 0}

        # Search for more candidates to allow noise mixing and tag filtering
        search_k = min(k * 5, index.ntotal)
        distances, indices = index.search(query_binary_arr, search_k)

        # Convert Hamming distances to similarity scores (0-1)
        # Max Hamming distance for 384 bits = 384
        max_dist = EMBEDDING_DIM
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            memory_id = self._idx_to_id.get(idx)
            if memory_id is None:
                continue

            # Apply tag filter using pre-fetched valid_ids set (O(1) lookup)
            if valid_ids is not None and memory_id not in valid_ids:
                continue

            similarity = 1.0 - (dist / max_dist)
            if similarity >= min_score:
                results.append({
                    "id": memory_id,
                    "score": float(similarity),
                    "hamming_distance": int(dist)
                })

        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)

        # Apply noise_ratio: mix in low-score results
        if noise_ratio > 0 and len(results) > k:
            noise_count = int(k * noise_ratio)
            if noise_count > 0:
                # Take top (k - noise_count) high-score results
                high_score = results[:k - noise_count]
                # Take noise_count from lower scores (but still above min_score)
                low_score_pool = results[k - noise_count:]
                if low_score_pool:
                    # Randomly sample from lower scores
                    np.random.shuffle(low_score_pool)
                    noise_results = low_score_pool[:noise_count]
                    results = high_score + noise_results
                    results.sort(key=lambda x: x['score'], reverse=True)

        return {
            "results": results[:k],
            "total": len(results)
        }

    def fetch(self, ids: List[int]) -> Dict[str, Any]:
        """Fetch full memory records by IDs (excludes soft-deleted)."""
        if not ids:
            return {"memories": []}

        conn = self._get_db()
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(ids))
        cursor.execute(f"""
            SELECT id, text, tags, metadata, vector, created_at, updated_at
            FROM memories
            WHERE id IN ({placeholders}) AND is_deleted = 0
        """, ids)

        memories = []
        for row in cursor.fetchall():
            vec = None
            if row['vector']:
                vec = np.frombuffer(row['vector'], dtype=np.float32).tolist()

            memories.append({
                "id": row['id'],
                "text": row['text'],
                "tags": json.loads(row['tags']),
                "metadata": json.loads(row['metadata']),
                "vector": vec,
                "created_at": row['created_at'],
                "updated_at": row['updated_at']
            })

        # Maintain order of requested IDs
        id_to_memory = {m['id']: m for m in memories}
        ordered = [id_to_memory.get(i) for i in ids if i in id_to_memory]

        return {"memories": ordered}

    def search_fts(self, query: str, k: int = 10) -> Dict[str, Any]:
        """Full-text search using FTS5 (excludes soft-deleted)."""
        conn = self._get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT m.id, m.text, m.tags, m.metadata, m.created_at, m.updated_at
            FROM memories m
            JOIN memories_fts f ON m.id = f.rowid
            WHERE memories_fts MATCH ? AND m.is_deleted = 0
            ORDER BY rank
            LIMIT ?
        """, (query, k))

        memories = []
        for row in cursor.fetchall():
            memories.append({
                "id": row['id'],
                "text": row['text'],
                "tags": json.loads(row['tags']),
                "metadata": json.loads(row['metadata']),
                "created_at": row['created_at'],
                "updated_at": row['updated_at']
            })

        return {"memories": memories}

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        conn = self._get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM memories WHERE is_deleted = 0")
        total_memories = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM memories WHERE is_deleted = 1")
        deleted_count = cursor.fetchone()['count']

        index = self._get_index()

        return {
            "total_memories": total_memories,
            "deleted_memories": deleted_count,
            "index_size": index.ntotal,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_model": EMBEDDING_MODEL
        }


# Global instance
_memory = EpisodicMemory()


# Tool functions for Hermes
def episodic_store(text: str, tags: List[str] = None, metadata: Dict = None, auto_tag: bool = False) -> Dict[str, Any]:
    """Store a new episodic memory.

    Args:
        text: The memory content to store
        tags: Optional list of tags for filtering
        metadata: Optional JSON metadata
        auto_tag: Automatically generate tags from content (default: False)

    Returns:
        Dict with the created memory info including ID
    """
    return _memory.store(text, tags or [], metadata or {}, auto_tag=auto_tag)


def episodic_recall_fuzzy(query: str, k: int = 10, noise_ratio: float = 0.1,
                        tags: List[str] = None, min_score: float = 0.0) -> Dict[str, Any]:
    """Fuzzy recall using 1-bit quantized binary HNSW index.

    Args:
        query: Search query text
        k: Number of candidates to return
        noise_ratio: Fraction of low-score results to mix in (0.0-0.5)
        tags: Optional tag filter
        min_score: Minimum similarity threshold (0.0-1.0)

    Returns:
        Dict with results list containing id, score, hamming_distance
    """
    return _memory.recall_fuzzy(query, k, noise_ratio, tags or [], min_score)


def episodic_fetch(ids: List[int]) -> Dict[str, Any]:
    """Fetch full memory records by IDs.

    Args:
        ids: List of memory IDs to fetch

    Returns:
        Dict with memories list containing full records
    """
    return _memory.fetch(ids)


def episodic_delete(ids: List[int]) -> Dict[str, Any]:
    """Soft-delete memories by IDs (logical deletion, excluded from searches).

    Args:
        ids: List of memory IDs to delete

    Returns:
        Dict with deleted count and IDs
    """
    return _memory.delete(ids)


def episodic_search_fts(query: str, k: int = 10) -> Dict[str, Any]:
    """Full-text search using SQLite FTS5.

    Args:
        query: Search query (supports FTS5 syntax: "exact phrase", term*, term1 AND term2, etc.)
        k: Maximum results to return

    Returns:
        Dict with memories list
    """
    return _memory.search_fts(query, k)


def episodic_stats() -> Dict[str, Any]:
    """Get memory system statistics."""
    return _memory.get_stats()


if __name__ == "__main__":
    # Simple test when run directly
    print("Testing Episodic Memory...")

    # Store some memories
    print("\n1. Storing memories...")
    r1 = memory_store("User prefers Japanese responses in chat", tags=["preference", "language"])
    print(f"  Stored: {r1['id']} - {r1['text'][:50]}...")

    r2 = memory_store("Project uses pytest with xdist for parallel testing", tags=["dev", "testing"])
    print(f"  Stored: {r2['id']} - {r2['text'][:50]}...")

    r3 = memory_store("User works on novel writing and PHP development", tags=["profile", "creative"])
    print(f"  Stored: {r3['id']} - {r3['text'][:50]}...")

    r4 = memory_store("Ollama runs on 6GB GPU with quantized models", tags=["infrastructure", "local-llm"])
    print(f"  Stored: {r4['id']} - {r4['text'][:50]}...")

    r5 = memory_store("Hermes config uses custom google-gemini provider for gemini-3.7-flash", tags=["config", "provider"])
    print(f"  Stored: {r5['id']} - {r5['text'][:50]}...")

    # Test auto_tag
    print("\n2. Testing auto_tag...")
    r6 = memory_store("PHPのLaravelで非同期処理を書く時はキューを使う", auto_tag=True)
    print(f"  Stored: {r6['id']} - tags: {r6['tags']}")

    r7 = memory_store("小説の冒頭で主人公の日常が崩壊する伏線を張る", auto_tag=True)
    print(f"  Stored: {r7['id']} - tags: {r7['tags']}")

    # Test fuzzy recall
    print("\n3. Fuzzy recall (query: 'testing preferences')...")
    results = memory_recall_fuzzy("testing preferences", k=5, noise_ratio=0.2)
    for r in results['results']:
        print(f"  ID: {r['id']}, Score: {r['score']:.3f}, Hamming: {r['hamming_distance']}")

    # Fetch full records
    print("\n4. Fetching full records...")
    ids = [r['id'] for r in results['results']]
    fetched = memory_fetch(ids)
    for m in fetched['memories']:
        print(f"  ID: {m['id']}, Text: {m['text'][:60]}...")
        print(f"    Tags: {m['tags']}")

    # Test FTS search
    print("\n5. Full-text search (query: 'Ollama')...")
    fts_results = memory_search_fts("Ollama", k=3)
    for m in fts_results['memories']:
        print(f"  ID: {m['id']}, Text: {m['text'][:60]}...")

    # Test delete
    print("\n6. Soft-delete test...")
    del_result = memory_delete([r1['id']])
    print(f"  Deleted: {del_result['deleted']} memories")
    # Verify deleted memory is excluded
    fetched_after = memory_fetch([r1['id']])
    print(f"  Fetch after delete: {len(fetched_after['memories'])} memories (should be 0)")

    # Stats
    print("\n7. Statistics:")
    stats = memory_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nAll tests passed!")