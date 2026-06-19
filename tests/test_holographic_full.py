import pytest
import tempfile
from pathlib import Path
from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever


class TestHolographicFull:
    def test_full_pipeline(self):
        """Test: complete add → retrieval pipeline"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MemoryStore(db_path=str(db_path))
            retriever = FactRetriever(store, onnx_weight=0.3)
            
            facts = [
                "deploy requires migration first",
                "run database migration before deploy",
                "python is a programming language",
                "javascript is used for web development",
            ]
            
            for fact in facts:
                store.add_fact(fact, category="test")
            
            results = retriever.search("deploy migration", category="test")
            
            assert len(results) >= 2
            deploy_scores = [
                r["score"] for r in results 
                if "deploy" in r["content"].lower() or "migration" in r["content"].lower()
            ]
            other_scores = [
                r["score"] for r in results 
                if "deploy" not in r["content"].lower() and "migration" not in r["content"].lower()
            ]
            
            if deploy_scores and other_scores:
                assert max(deploy_scores) > max(other_scores)
            
            store.close()
