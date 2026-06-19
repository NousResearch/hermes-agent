import pytest
import tempfile
from pathlib import Path
from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever


class TestRetrievalFusion:
    def test_search_with_onnx_weight(self):
        """Test: search accepts onnx_weight parameter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MemoryStore(db_path=str(db_path))
            retriever = FactRetriever(store, onnx_weight=0.3)
            
            store.add_fact("deploy requires migrations first", category="test")
            store.add_fact("run database migration before deploy", category="test")
            
            results = retriever.search("deploy migration", category="test")
            
            assert len(results) > 0
            assert "score" in results[0]
            
            store.close()
    
    def test_onnx_similarity_computation(self):
        """Test: ONNX similarity is computed and affects ranking"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MemoryStore(db_path=str(db_path))
            
            import numpy as np
            
            fact1_id = store.add_fact("deploy requires migration first", category="test")
            fact2_id = store.add_fact("python deploy migration guide", category="test")
            
            query_embedding = np.random.rand(768).astype(np.float32)
            similar_embedding = query_embedding + np.random.normal(0, 0.1, 768).astype(np.float32)
            different_embedding = np.random.rand(768).astype(np.float32)
            
            store._conn.execute(
                "UPDATE facts SET embedding = ? WHERE fact_id = ?",
                (similar_embedding.tobytes(), fact1_id)
            )
            store._conn.execute(
                "UPDATE facts SET embedding = ? WHERE fact_id = ?",
                (different_embedding.tobytes(), fact2_id)
            )
            store._conn.commit()
            
            retriever = FactRetriever(store, onnx_weight=0.8, fts_weight=0.1, jaccard_weight=0.1, hrr_weight=0.0)
            
            class MockEmbedder:
                def embed(self, text):
                    return query_embedding
            
            import plugins.memory.holographic.retrieval as retrieval_module
            original_get_embedder = retrieval_module.get_embedder
            retrieval_module.get_embedder = lambda: MockEmbedder()
            
            try:
                results = retriever.search("deploy migration", category="test")
                
                assert len(results) == 2
                assert results[0]["fact_id"] == fact1_id
                assert results[0]["score"] > results[1]["score"]
            finally:
                retrieval_module.get_embedder = original_get_embedder
            
            store.close()
