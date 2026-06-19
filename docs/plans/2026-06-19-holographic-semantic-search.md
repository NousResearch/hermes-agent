# Holographic 语义搜索增强实现计划

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 在 holographic 插件中添加 ONNX 语义搜索能力，形成 FTS5 + Jaccard + HRR + ONNX 四路融合检索。

**Architecture:** 在现有 holographic 插件基础上，添加 ONNX 嵌入模块（nomic-embed-text-v1.5），在 SQLite 中为每个 fact 存储语义向量，检索时计算查询与所有 fact 的余弦相似度，与现有三路检索结果加权融合。

**Tech Stack:** Python 3.12, SQLite, ONNX Runtime, numpy, nomic-embed-text-v1.5 (768-dim)

---

## 前置条件

- ONNX 模型已缓存: `~/.aingram/models/nomic-embed-text-v1.5/onnx/model.onnx` (523MB)
- Tokenizer 已缓存: `~/.aingram/models/nomic-embed-text-v1.5/tokenizer.json`
- holographic 插件路径: `~/.hermes/hermes-agent/plugins/memory/holographic/`
- 测试路径: `~/.hermes/hermes-agent/tests/`
- 依赖: `onnxruntime`, `tokenizers`, `numpy`

---

## Task 1: 创建 ONNX 嵌入模块 (embedder.py)

**Objective:** 创建 `embedder.py`，封装 ONNX 模型加载、文本嵌入生成、余弦相似度计算，并提供优雅降级。

**Files:**
- Create: `plugins/memory/holographic/embedder.py`
- Test: `tests/test_embedder.py`

### Step 1: 写失败测试

```python
# tests/test_embedder.py
import pytest
from pathlib import Path

# 模型路径
MODEL_PATH = Path("~/.aingram/models/nomic-embed-text-v1.5/onnx/model.onnx").expanduser()


class TestEmbedder:
    """测试 ONNX 嵌入模块"""

    def test_import_without_onnx(self):
        """测试：导入模块不会失败（即使 ONNX 不可用）"""
        from plugins.memory.holographic.embedder import get_embedder, cosine_similarity
        assert get_embedder is not None
        assert cosine_similarity is not None

    def test_cosine_similarity_identical(self):
        """测试：相同向量的余弦相似度为 1.0"""
        from plugins.memory.holographic.embedder import cosine_similarity
        import numpy as np
        vec = np.random.rand(768).astype(np.float32)
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """测试：正交向量的余弦相似度接近 0.0"""
        from plugins.memory.holographic.embedder import cosine_similarity
        import numpy as np
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim) < 1e-6

    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="ONNX 模型不可用")
    def test_embed_returns_vector(self):
        """测试：嵌入生成返回正确维度的向量"""
        from plugins.memory.holographic.embedder import get_embedder
        embedder = get_embedder()
        if embedder is None:
            pytest.skip("ONNX model not available")
        vec = embedder.embed("hello world")
        assert vec is not None
        assert vec.shape == (768,)
        assert vec.dtype.name == "float32"

    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="ONNX 模型不可用")
    def test_embed_empty_string(self):
        """测试：空字符串嵌入返回 None"""
        from plugins.memory.holographic.embedder import get_embedder
        embedder = get_embedder()
        if embedder is None:
            pytest.skip("ONNX model not available")
        vec = embedder.embed("")
        assert vec is None
```

### Step 2: 运行测试验证失败

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/test_embedder.py -v`
Expected: FAIL — "ModuleNotFoundError: No module named 'holographic.embedder'"

### Step 3: 写最小实现

```python
# plugins/memory/holographic/embedder.py
"""ONNX embedding module for semantic search.

Wraps nomic-embed-text-v1.5 ONNX model for text embedding generation.
Provides graceful degradation when model is unavailable.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model path
_MODEL_PATH = Path("~/.aingram/models/nomic-embed-text-v1.5/onnx/model.onnx").expanduser()
_EMBED_DIM = 768


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Returns:
        Float in [-1, 1]. 1.0 for identical, 0.0 for orthogonal.
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


class OnnxEmbedder:
    """ONNX-based text embedder using nomic-embed-text-v1.5."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self._model_path = model_path or _MODEL_PATH
        self._session = None
        self._available = False
        
        if self._model_path.exists():
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(str(self._model_path))
                self._available = True
                logger.info("ONNX embedder loaded: %s", self._model_path)
            except Exception as e:
                logger.warning("Failed to load ONNX model: %s", e)
        else:
            logger.warning("ONNX model not found: %s", self._model_path)
    
    @property
    def available(self) -> bool:
        """Check if embedder is ready."""
        return self._available
    
    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return _EMBED_DIM
    
    def embed(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text.
        
        Args:
            text: Input text to embed.
            
        Returns:
            np.ndarray of shape (768,) with float32 values, or None if unavailable.
        """
        if not self._available or not text.strip():
            return None
        
        try:
            # Tokenize using loaded tokenizer
            tokens = self._tokenize(text)
            
            # Run inference
            input_ids = np.array([tokens], dtype=np.int64)
            attention_mask = np.ones_like(input_ids)
            
            outputs = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )
            
            # Mean pooling
            embedding = outputs[0].mean(axis=1).squeeze()
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            return None
    
    def _load_tokenizer(self) -> None:
        """Load tokenizer from tokenizer.json using tokenizers library.
        
        Raises:
            FileNotFoundError: If tokenizer.json not found.
            ImportError: If tokenizers library not installed.
        """
        tokenizer_path = self._model_path.parent / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        try:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except ImportError:
            raise ImportError("tokenizers library required: pip install tokenizers")
    
    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text using loaded tokenizer.
        
        Args:
            text: Input text.
            
        Returns:
            List of token IDs.
        """
        if not hasattr(self, "_tokenizer"):
            self._load_tokenizer()
        
        # Use tokenizers library for proper WordPiece tokenization
        encoding = self._tokenizer.encode(text)
        return encoding.ids


# Singleton instance
_embedder: Optional[OnnxEmbedder] = None


def get_embedder() -> Optional[OnnxEmbedder]:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = OnnxEmbedder()
    return _embedder if _embedder.available else None
```

### Step 4: 运行测试验证通过

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/test_embedder.py -v`
Expected: PASS

### Step 5: 提交

```bash
cd ~/.hermes/hermes-agent
git add plugins/memory/holographic/embedder.py tests/test_embedder.py
git commit -m "feat(holographic): add ONNX embedder module"
```

**Acceptance Criteria:**
1. `from holographic.embedder import get_embedder, cosine_similarity` 成功
2. `cosine_similarity(vec, vec)` 返回 1.0（误差 < 1e-6）
3. `get_embedder()` 在模型不可用时返回 None（不抛异常）
4. `pytest tests/test_embedder.py -v` 显示 5 passed
5. No regression in holographic plugin tests (if any exist)

---

## Task 2: 修改 store.py 添加 embedding 列

**Objective:** 在 facts 表添加 embedding BLOB 列，add_fact() 时计算 ONNX embedding，提供 backfill_existing_facts() 批量补算。

**Files:**
- Modify: `plugins/memory/holographic/store.py:128-140` (ALTER TABLE)
- Modify: `plugins/memory/holographic/store.py:146-189` (add_fact)
- Create: `tests/test_store_embedding.py`

### Step 1: 写失败测试

```python
# tests/test_store_embedding.py
import pytest
import tempfile
from pathlib import Path
from plugins.memory.holographic.store import MemoryStore


class TestStoreEmbedding:
    """测试 store 的 embedding 功能"""
    
    def test_add_fact_with_embedding(self):
        """测试：add_fact 时计算 embedding"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MemoryStore(db_path=str(db_path))
            
            # 添加 fact
            fact_id = store.add_fact("test content", category="test")
            
            # 验证 embedding 列存在
            row = store._conn.execute(
                "SELECT embedding FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            assert row is not None
            assert "embedding" in row.keys()
            
            store.close()
    
    def test_backfill_existing_facts(self):
        """测试：backfill_existing_facts 补算旧 fact 的 embedding"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MemoryStore(db_path=str(db_path))
            
            # 手动插入一个没有 embedding 的 fact
            store._conn.execute(
                "INSERT INTO facts (content, category) VALUES (?, ?)",
                ("old fact", "test")
            )
            store._conn.commit()
            
            # 运行 backfill
            count = store.backfill_existing_facts()
            
            # 验证
            assert count == 1
            row = store._conn.execute(
                "SELECT embedding FROM facts WHERE content = ?", ("old fact",)
            ).fetchone()
            assert row["embedding"] is not None
            
            store.close()
```

### Step 2: 运行测试验证失败

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/test_store_embedding.py -v`
Expected: FAIL — "sqlite3.OperationalError: no such column: embedding"

### Step 3: 写最小实现

**Diff 1: 添加 embedding 列迁移 (store.py:128-140)**

```python
# INSIDE _init_db(), after hrr_vector migration (line 140):
        # Migrate: add embedding column if missing (ONNX semantic vectors)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "embedding" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN embedding BLOB")
        self._conn.commit()
```

**Diff 2: 修改 add_fact 计算 embedding (store.py:146-189)**

```python
# ADD import at top of store.py:
from .embedder import get_embedder

# INSIDE add_fact(), after _compute_hrr_vector (line 186):
            # Compute ONNX embedding for semantic search
            self._compute_embedding(fact_id, content)
```

**Diff 3: 添加 _compute_embedding 方法**

```python
# ADD new method after _compute_hrr_vector (line 497):
    def _compute_embedding(self, fact_id: int, content: str) -> None:
        """Compute and store ONNX embedding for a fact. No-op if unavailable."""
        with self._lock:
            embedder = get_embedder()
            if embedder is None:
                return
            
            embedding = embedder.embed(content)
            if embedding is None:
                return
            
            # Serialize to bytes (float32 → bytes)
            self._conn.execute(
                "UPDATE facts SET embedding = ? WHERE fact_id = ?",
                (embedding.tobytes(), fact_id),
            )
            self._conn.commit()
```

**Diff 4: 添加 backfill_existing_facts 方法**

```python
# ADD new method after rebuild_all_vectors (line 560):
    
    # Also modify rebuild_all_vectors to include ONNX embeddings:
    # Inside rebuild_all_vectors(), after _compute_hrr_vector call, add:
    #     self._compute_embedding(row["fact_id"], row["content"])
    
    def backfill_existing_facts(self, batch_size: int = 100) -> int:
        """Compute ONNX embeddings for all facts missing them.
        
        Args:
            batch_size: Number of facts to process per commit.
        
        Returns:
            Number of facts processed.
        """
        embedder = get_embedder()
        if embedder is None:
            return 0
        
        # Get facts without embeddings (outside lock)
        rows = self._conn.execute(
            "SELECT fact_id, content FROM facts WHERE embedding IS NULL"
        ).fetchall()
        
        count = 0
        for i, row in enumerate(rows):
            # Compute embedding outside lock
            embedding = embedder.embed(row["content"])
            
            # Update inside lock (brief)
            if embedding is not None:
                with self._lock:
                    self._conn.execute(
                        "UPDATE facts SET embedding = ? WHERE fact_id = ?",
                        (embedding.tobytes(), row["fact_id"]),
                    )
                    count += 1
            
            # Commit in batches
            if (i + 1) % batch_size == 0:
                with self._lock:
                    self._conn.commit()
        
        # Final commit
        with self._lock:
            self._conn.commit()
        
        return count
```

### Step 4: 运行测试验证通过

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/test_store_embedding.py -v`
Expected: PASS

### Step 5: 提交

```bash
cd ~/.hermes/hermes-agent
git add plugins/memory/holographic/store.py tests/test_store_embedding.py
git commit -m "feat(holographic): add embedding column and backfill"
```

**Acceptance Criteria:**
1. `ALTER TABLE facts ADD COLUMN embedding BLOB` 成功执行
2. `add_fact()` 后 embedding 列有值（如果 ONNX 可用）
3. `backfill_existing_facts()` 补算所有旧 fact 的 embedding
4. `pytest tests/test_store_embedding.py -v` 显示 2 passed
5. No regression in holographic plugin tests (if any exist)

---

## Task 3: 修改 retrieval.py 添加第四路融合

**Objective:** 在 FactRetriever.search() 中添加 ONNX 语义相似度作为第四路检索信号。

**Files:**
- Modify: `plugins/memory/holographic/retrieval.py:22-112` (FactRetriever)
- Create: `tests/test_retrieval_fusion.py`

### Step 1: 写失败测试

```python
# tests/test_retrieval_fusion.py
import pytest
import tempfile
from pathlib import Path
from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever


class TestRetrievalFusion:
    """测试四路融合检索"""
    
    def test_search_with_onnx_weight(self):
        """Test: search accepts onnx_weight parameter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MemoryStore(db_path=str(db_path))
            retriever = FactRetriever(store, onnx_weight=0.3)
            
            # Add test data
            store.add_fact("deploy requires migrations first", category="test")
            store.add_fact("run database migration before deploy", category="test")
            
            # Search
            results = retriever.search("deploy migration", category="test")
            
            # Verify results contain score
            assert len(results) > 0
            assert "score" in results[0]
            
            store.close()
    
    def test_onnx_similarity_computation(self):
        """Test: ONNX similarity is computed and affects ranking"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MemoryStore(db_path=str(db_path))
            
            # Insert facts with known embeddings
            import numpy as np
            from plugins.memory.holographic.embedder import cosine_similarity
            
            # Create two facts
            fact1_id = store.add_fact("deploy requires migrations first", category="test")
            fact2_id = store.add_fact("python is a programming language", category="test")
            
            # Mock embeddings: similar to "deploy migration" query
            query_embedding = np.random.rand(768).astype(np.float32)
            similar_embedding = query_embedding + np.random.normal(0, 0.1, 768).astype(np.float32)
            different_embedding = np.random.rand(768).astype(np.float32)
            
            # Store mock embeddings
            store._conn.execute(
                "UPDATE facts SET embedding = ? WHERE fact_id = ?",
                (similar_embedding.tobytes(), fact1_id)
            )
            store._conn.execute(
                "UPDATE facts SET embedding = ? WHERE fact_id = ?",
                (different_embedding.tobytes(), fact2_id)
            )
            store._conn.commit()
            
            # Create retriever with high ONNX weight
            retriever = FactRetriever(store, onnx_weight=0.8, fts_weight=0.1, jaccard_weight=0.1, hrr_weight=0.0)
            
            # Mock embedder to return our known query embedding
            class MockEmbedder:
                def embed(self, text):
                    return query_embedding
            
            # Patch get_embedder to return mock
            import plugins.memory.holographic.retrieval as retrieval_module
            original_get_embedder = retrieval_module.get_embedder
            retrieval_module.get_embedder = lambda: MockEmbedder()
            
            try:
                # Search
                results = retriever.search("deploy migration", category="test")
                
                # Verify similar fact ranks higher
                assert len(results) == 2
                assert results[0]["fact_id"] == fact1_id  # Similar embedding should rank first
                assert results[0]["score"] > results[1]["score"]
            finally:
                # Restore original
                retrieval_module.get_embedder = original_get_embedder
            
            store.close()
```

### Step 2: 运行测试验证失败

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/test_retrieval_fusion.py -v`
Expected: FAIL — "TypeError: __init__() got an unexpected keyword argument 'onnx_weight'"

### Step 3: 写最小实现

**Diff 1: 修改 __init__ 添加 onnx_weight (retrieval.py:22-47)**

```python
# MODIFY __init__ signature (line 22-33):
    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.4,
        jaccard_weight: float = 0.3,
        hrr_weight: float = 0.3,
        onnx_weight: float = 0.2,  # NEW: ONNX semantic weight (total weights sum to 1.2)
        hrr_dim: int = 1024,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # Auto-redistribute weights if numpy unavailable
        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            fts_weight = 0.5
            jaccard_weight = 0.5
            hrr_weight = 0.0
            onnx_weight = 0.0
        
        # Normalize weights to sum to 1.0
        total_weight = fts_weight + jaccard_weight + hrr_weight + onnx_weight
        if total_weight > 0:
            self.fts_weight = fts_weight / total_weight
            self.jaccard_weight = jaccard_weight / total_weight
            self.hrr_weight = hrr_weight / total_weight
            self.onnx_weight = onnx_weight / total_weight
        else:
            # Fallback to equal weights
            self.fts_weight = 0.25
            self.jaccard_weight = 0.25
            self.hrr_weight = 0.25
            self.onnx_weight = 0.25

        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight
        self.onnx_weight = onnx_weight
```

**Diff 2: 修改 search 方法添加 ONNX 相似度 (retrieval.py:48-112)**

```python
# ADD import at top of retrieval.py:
from .embedder import get_embedder, cosine_similarity

# INSIDE search(), after HRR similarity computation (line 89):
            # ONNX semantic similarity
            if self.onnx_weight > 0 and fact.get("embedding"):
                from .store import MemoryStore
                from .embedder import get_embedder, cosine_similarity
                import numpy as np
                
                embedder = get_embedder()
                if embedder is not None:
                    # Deserialize fact embedding
                    fact_embedding = np.frombuffer(fact["embedding"], dtype=np.float32)
                    
                    # Compute query embedding
                    query_embedding = embedder.embed(query)
                    
                    if query_embedding is not None:
                        onnx_sim = cosine_similarity(query_embedding, fact_embedding)
                        # Shift from [-1,1] to [0,1]
                        onnx_sim = (onnx_sim + 1.0) / 2.0
                    else:
                        onnx_sim = 0.5  # neutral
                else:
                    onnx_sim = 0.5  # neutral
            else:
                onnx_sim = 0.5  # neutral
```

**Diff 3: 修改 combine 公式 (retrieval.py:91-94)**

```python
# MODIFY combine formula (line 91-94):
            # Combine FTS5 + Jaccard + HRR + ONNX
            relevance = (self.fts_weight * fts_score
                        + self.jaccard_weight * jaccard
                        + self.hrr_weight * hrr_sim
                        + self.onnx_weight * onnx_sim)
```

**Diff 4: 在返回前清理 embedding 字段 (retrieval.py:109-112)**

```python
# MODIFY strip raw bytes (line 109-112):
        # Strip raw bytes — callers expect JSON-serializable dicts
        for fact in results:
            fact.pop("hrr_vector", None)
            fact.pop("embedding", None)  # ADD: also strip embedding
```

### Step 4: 运行测试验证通过

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/test_retrieval_fusion.py -v`
Expected: PASS

### Step 5: 提交

```bash
cd ~/.hermes/hermes-agent
git add plugins/memory/holographic/retrieval.py tests/test_retrieval_fusion.py
git commit -m "feat(holographic): add ONNX semantic fusion to retrieval"
```

**Acceptance Criteria:**
1. `FactRetriever(store, onnx_weight=0.3)` 创建成功
2. `search()` 返回结果包含 ONNX 相似度贡献
3. 四路权重正确：fts_weight + jaccard_weight + hrr_weight + onnx_weight
4. `pytest tests/test_retrieval_fusion.py -v` 显示 2 passed
5. No regression in holographic plugin tests (if any exist)

---

## Task 4: 集成测试和文档更新

**Objective:** 运行完整测试套件，更新 CHANGELOG 和相关文档。

**Files:**
- Test: `tests/test_holographic_full.py`
- Modify: `CHANGELOG.md`
- Modify: `docs/technical/ARCHITECTURE.md` (if exists)

### Step 1: 写集成测试

```python
# tests/test_holographic_full.py
import pytest
import tempfile
from pathlib import Path
from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever


class TestHolographicFull:
    """端到端集成测试"""
    
    def test_full_pipeline(self):
        """Test: complete add → retrieval pipeline"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MemoryStore(db_path=str(db_path))
            retriever = FactRetriever(store, onnx_weight=0.3)
            
            # Add multiple related facts
            facts = [
                "deploy requires migrations first",
                "run database migration before deploy",
                "python is a programming language",
                "javascript is used for web development",
            ]
            
            for fact in facts:
                store.add_fact(fact, category="test")
            
            # Search for deploy-related
            results = retriever.search("deploy migration", category="test")
            
            # Verify ranking makes sense (deploy-related should rank higher)
            assert len(results) >= 2
            deploy_scores = [
                r["score"] for r in results 
                if "deploy" in r["content"].lower() or "migration" in r["content"].lower()
            ]
            other_scores = [
                r["score"] for r in results 
                if "deploy" not in r["content"].lower() and "migration" not in r["content"].lower()
            ]
            
            # Deploy-related should score higher
            if deploy_scores and other_scores:
                assert max(deploy_scores) > max(other_scores)
            
            store.close()
```

### Step 2: 运行集成测试

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/test_holographic_full.py -v`
Expected: PASS

### Step 3: 运行 holographic 相关测试

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/test_embedder.py tests/test_store_embedding.py tests/test_retrieval_fusion.py tests/test_holographic_full.py -v`
Expected: All holographic tests pass

### Step 4: 更新文档（如果存在）

```bash
# Check if CHANGELOG.md exists
if [ -f CHANGELOG.md ]; then
  echo "CHANGELOG.md exists, adding entry..."
  # Add entry to CHANGELOG.md
fi

# Check if docs/technical/ARCHITECTURE.md exists
if [ -f docs/technical/ARCHITECTURE.md ]; then
  echo "ARCHITECTURE.md exists, updating..."
  # Update architecture docs
fi
```

**Note:** Only update documentation files that exist. Do not create new documentation files as part of this task.

### Step 5: 提交

```bash
cd ~/.hermes/hermes-agent
git add tests/test_holographic_full.py CHANGELOG.md
git commit -m "test(holographic): add integration tests and update changelog"
```

**Acceptance Criteria:**
1. `pytest tests/ -v` 全部通过
2. CHANGELOG.md 有新条目
3. 集成测试验证四路融合效果
4. 无安全问题（无硬编码 secrets）
5. 文档与代码同步

---

## 执行顺序

```
Task 1 (embedder.py) → Task 2 (store.py) → Task 3 (retrieval.py) → Task 4 (集成测试)
```

每个 Task 完成后提交，确保可断点续传。

---

## 验证命令

```bash
# 运行所有 holographic 相关测试
cd ~/.hermes/hermes-agent
python -m pytest tests/test_embedder.py tests/test_store_embedding.py tests/test_retrieval_fusion.py tests/test_holographic_full.py -v

# 运行完整测试套件
python -m pytest tests/ -v

# Check code quality (if ruff available)
python -m ruff check plugins/memory/holographic/ 2>/dev/null || echo "ruff not installed"
```

---

## 注意事项

1. **ONNX 模型降级**: 如果 ONNX 模型不可用，所有功能正常工作，只是 ONNX 相似度为 0.5（中性）
2. **性能影响**: 首次加载 ONNX 模型约 1.3s，后续嵌入约 30ms
3. **内存开销**: ONNX Runtime + 模型约 100MB，2C/4G VPS 可承受
4. **向后兼容**: 现有三路检索不受影响，权重可配置
5. **数据迁移**: `backfill_existing_facts()` 可批量补算旧 fact 的 embedding

---

**计划状态:** 待审查 (Gate 2)

---

## 审查结果: REQUEST_CHANGES

### Checklist

| # | Criteria | Status | Notes |
|---|----------|--------|-------|
| 1 | Task granularity (2-5 min) | ✅ PASS | Each task is well-scoped for 2-5 min execution |
| 2 | File paths (exact) | ⚠️ ISSUE | Line number ranges will drift after first edit; use method names or sentinel comments instead |
| 3 | Code examples (complete) | ❌ FAIL | `_simple_tokenize` in embedder.py is non-functional placeholder; test imports use wrong package path |
| 4 | Commands (exact with expected output) | ✅ PASS | All commands are precise with expected output documented |
| 5 | TDD (test first) | ✅ PASS | Every task follows test-first pattern correctly |
| 6 | Verification steps | ❌ FAIL | Task 3 tests only verify parameter wiring, not actual ONNX similarity computation |
| 7 | DRY | ✅ PASS | No unnecessary repetition |
| 8 | YAGNI | ✅ PASS | Minimal, focused scope |
| 9 | Missing context | ❌ FAIL | Tokenizer dependency not mentioned; no conftest/PYTHONPATH strategy for test imports |
| 10 | Backward compatible | ✅ PASS | ALTER TABLE ADD COLUMN, default params, field stripping all safe |
| 11 | Dependencies (task order) | ✅ PASS | Task order is correct |
| 12 | Integration | ⚠️ ISSUE | Import style inconsistent with existing modules; weight sum > 1.0 changes score magnitude |

### Critical Issues (Must Fix Before Approval)

**Issue #1: Test import paths are wrong for the project's package structure**

The test files use `from plugins.memory.holographic.store import MemoryStore`, but the module lives at `plugins/memory/holographic/store.py`. With the project root on `sys.path` (set by `tests/conftest.py`), the correct import is:

```python
from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.embedder import get_embedder, cosine_similarity
```

Every test file in the plan (test_embedder.py, test_store_embedding.py, test_retrieval_fusion.py, test_holographic_full.py) has this bug. Running them will fail with `ModuleNotFoundError: No module named 'holographic'`.

**Issue #2: `_simple_tokenize` in embedder.py is non-functional**

```python
def _simple_tokenize(self, text: str) -> list[int]:
    return [101] + [ord(c) % 30000 for c in text[:512]] + [102]
```

The nomic-embed-text-v1.5 ONNX model expects token IDs from a proper BPE tokenizer (tokenizer.json). Using `ord(c) % 30000` produces garbage token IDs that will yield meaningless embeddings. The code itself acknowledges this with "This is a simplified version — real implementation would load tokenizer.json" but no follow-up task provides it. With a real model path (~523MB), `test_embed_returns_vector` would pass shape checks but produce semantically meaningless vectors, making the entire four-way fusion deliver noise. **The plan must include proper tokenizer integration or document the dependency explicitly.**

### High Issues

**Issue #3: Task 3 tests don't verify actual ONNX similarity computation**

`test_search_with_onnx_weight` only checks that `onnx_weight=0.3` is accepted and search returns results with scores. `test_onnx_similarity_computation` only checks `retriever.onnx_weight == 0.5`. Neither test actually verifies that:
- Cosine similarity is computed between query embedding and fact embedding
- The ONNX score is incorporated into the final relevance score
- Semantic similarity (e.g., "deploy" query ranks "deploy migration" above "python programming")

**Issue #4: No tokenizer dependency documented**

The plan lists the ONNX model path (`model.onnx`, 523MB) but doesn't mention the required `tokenizer.json` (or `tokenizer_config.json`) that must accompany the model. An ONNX model file alone is insufficient for text embedding — the tokenizer vocabulary and configuration are required to convert text to input IDs. Without this, an implementer following the plan verbatim will get a working `embedder.py` that produces garbage embeddings.

**Issue #5: `backfill_existing_facts` holds database lock during slow ONNX inference**

```python
def backfill_existing_facts(self) -> int:
    with self._lock:  # ← held for potentially hundreds of ~30ms embedding calls
        ...
        for row in rows:
            embedding = embedder.embed(row["content"])
```

For N=500 facts at ~30ms each, this blocks all DB operations for ~15 seconds. In a single-user CLI agent this may be acceptable, but the pattern is fragile. The embedding loop should be moved outside the lock, or the lock should be scoped to individual UPDATE statements.

### Medium Issues

**Issue #6: Line number ranges will drift**

The plan references exact line ranges (e.g., `store.py:128-140`, `retrieval.py:22-112`). After the first diff is applied, every subsequent line number is wrong. The plan should reference method names or use sentinel comments (`# ADD: embedding column migration`).

**Issue #7: Weight sum exceeds 1.0 without normalization**

Existing default weights: fts=0.4, jaccard=0.3, hrr=0.3 → sum=1.0. Adding onnx_weight=0.3 gives sum=1.3. The retrieval code doesn't normalize weights, so the ONNX contribution is disproportionately represented relative to existing signals. Either normalize or document that weights are relative proportions.

**Issue #8: "Existing tests no regression" is misleading**

The plan repeatedly states "现有测试无回归" (no regression in existing tests) as an acceptance criterion. There are **zero** existing holographic plugin tests. Running `pytest tests/ -v` would test the entire project (~50 test files), and failures in unrelated tests would block the integration task. This criterion should either be removed or clarified.

### Low Issues

**Issue #9: Chinese test docstrings in an English codebase**

The codebase consistently uses English for docstrings and comments. The plan's Chinese test docstrings (`"""测试：..."""`) are inconsistent with project conventions.

**Issue #10: Import style inconsistency**

Existing modules use try/except patterns for optional dependencies:
```python
try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr
```

The new `embedder.py` uses a direct `import onnxruntime as ort` inside methods. For consistency, ONNX runtime import should use the same try/except guard pattern. The current code will crash at import-adjacent time rather than degrade gracefully if onnxruntime is not installed.

### Acceptance Criteria (Per Task)

**Task 1: Create embedder.py**
1. ✅ `from plugins.memory.holographic.embedder import get_embedder, cosine_similarity` succeeds
2. ✅ `cosine_similarity(vec, vec)` returns 1.0 (within 1e-6)
3. ✅ `cosine_similarity(orthogonal_vecs)` returns ~0.0
4. ✅ `get_embedder()` returns `None` when model/tokenizer unavailable (no exception)
5. ❌ `embedder.embed("hello world")` returns a valid 768-dim float32 vector — **blocked by Issue #2 (non-functional tokenizer)**

**Task 2: Modify store.py**
1. ✅ `ALTER TABLE facts ADD COLUMN embedding BLOB` executes on existing databases
2. ✅ `add_fact()` computes and stores ONNX embedding (no-op when unavailable)
3. ✅ `backfill_existing_facts()` processes all facts with NULL embedding
4. ⚠️ `backfill_existing_facts()` releases lock during embedding computation — **Issue #5**
5. ✅ Existing `_compute_hrr_vector` and `rebuild_all_vectors` continue working unchanged

**Task 3: Modify retrieval.py**
1. ✅ `FactRetriever(store, onnx_weight=0.3)` creates successfully with default backward compatibility
2. ⚠️ `search()` computes ONNX cosine similarity and incorporates it into final score — **Issue #3 (no verification test)**
3. ✅ `onnx_weight=0.0` disables ONNX path entirely
4. ✅ `embedding` field is stripped from returned results (JSON-safe)
5. ❌ Existing callers without `onnx_weight` parameter still work — **Issue #7 (weight sum > 1.0 changes behavior)**

**Task 4: Integration test**
1. ❌ `pytest tests/ -v` passes — **blocked by Issue #1 (import paths) and potential unrelated test failures**
2. ✅ CHANGELOG.md has new entry
3. ⚠️ End-to-end test verifies semantic ranking (deploy-related facts score higher) — **blocked by Issue #2 (broken tokenizer)**
4. ✅ No hardcoded secrets or security issues
5. ✅ Documentation changes synchronized with code

### Verdict

**REQUEST_CHANGES** — Two critical issues (wrong test import paths, non-functional tokenizer) block implementation. Two high issues (weak verification, missing tokenizer dependency) must be resolved before the plan can deliver working semantic search. Recommend fixing and re-reviewing before assigning to an implementer.