"""Tests for soma.context_builder — three blocks, caps, similarity filter."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from soma.context_builder import BEHAVIOR_TAGS, ContextBuilder
from soma.memory_store import MemoryStore


class _KeyedEmbedder:
    """Embedder that hands out unit vectors. Identical text → identical vector;
    different text → orthogonal vector. Lets tests precisely control what
    query()'s cosine sees."""

    def __init__(self):
        self._slots: dict[str, int] = {}
        self._dim = 64

    def embed(self, text: str) -> list[float]:
        key = text.strip().lower()
        if key not in self._slots:
            self._slots[key] = len(self._slots)
        vec = [0.0] * self._dim
        vec[self._slots[key] % self._dim] = 1.0
        return vec


class ContextBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self.tmp.close()
        self.addCleanup(lambda: os.path.exists(self.tmp.name) and os.unlink(self.tmp.name))
        self.embedder = _KeyedEmbedder()
        self.store = MemoryStore(self.tmp.name, self.embedder)

    def test_empty_store_returns_empty_text(self):
        builder = ContextBuilder(self.store)
        result = builder.build("anything")
        self.assertEqual(result.text, "")
        self.assertEqual(result.behavior_count, 0)

    def test_behavior_rules_always_present(self):
        self.store.write("Answer in German.", type="semantic", tags=["preference"])
        self.store.write("Never run rm without confirmation.", type="procedural", tags=["behavior"])
        # Query unrelated to either rule.
        result = ContextBuilder(self.store).build("totally unrelated query")
        self.assertEqual(result.behavior_count, 2)
        self.assertIn("BEHAVIOR RULES", result.text)
        self.assertIn("Answer in German.", result.text)
        self.assertIn("Never run rm without confirmation.", result.text)

    def test_relevant_context_block_uses_query(self):
        self.store.write("User works at Supermicro", type="semantic", tags=["domain"])
        self.store.write("User likes hiking", type="semantic", tags=["hobby"])
        # Embedder gives identical vectors for identical lowercased text.
        result = ContextBuilder(self.store, min_similarity=0.5).build(
            "user works at supermicro"
        )
        self.assertIn("RELEVANT CONTEXT", result.text)
        self.assertIn("User works at Supermicro", result.text)
        self.assertNotIn("User likes hiking", result.text)

    def test_procedures_block_only_procedural(self):
        self.store.write("do the dance", type="procedural", tags=[])
        self.store.write("do the dance fact", type="semantic", tags=[])
        result = ContextBuilder(self.store, min_similarity=0.0).build("do the dance")
        self.assertIn("PROCEDURES", result.text)
        # The procedural memory appears under PROCEDURES.
        procs = result.text.split("PROCEDURES", 1)[1]
        self.assertIn("do the dance", procs)
        # The semantic memory appears under RELEVANT CONTEXT, not duplicated below.
        self.assertEqual(procs.count("do the dance fact"), 0)

    def test_behavior_excluded_from_context_block(self):
        self.store.write("Be concise.", type="semantic", tags=["preference"])
        result = ContextBuilder(self.store, min_similarity=0.0).build("Be concise.")
        # Appears in BEHAVIOR RULES exactly once — not duplicated under RELEVANT CONTEXT.
        self.assertEqual(result.text.count("Be concise."), 1)

    def test_caps_per_block(self):
        for i in range(20):
            self.store.write(f"behavior rule {i}", type="semantic", tags=["behavior"])
        result = ContextBuilder(self.store, max_behavior=5).build("anything")
        self.assertEqual(result.behavior_count, 5)

    def test_min_similarity_filters_irrelevant(self):
        self.store.write("totally unrelated fact", type="semantic", tags=[])
        # Query is orthogonal to the stored memory → similarity 0 → filtered out.
        result = ContextBuilder(self.store, min_similarity=0.5).build("different query")
        self.assertEqual(result.context_count, 0)

    def test_curator_notes_block(self):
        result = ContextBuilder(self.store).build(
            "q", curator_notes=["found new doc about X"]
        )
        self.assertIn("CURATOR NOTES", result.text)
        self.assertIn("found new doc about X", result.text)

    def test_header_appears_when_content_present(self):
        self.store.write("rule", type="semantic", tags=["behavior"])
        result = ContextBuilder(self.store).build("q")
        self.assertIn("Soma context", result.text)

    def test_behavior_tag_constants(self):
        self.assertEqual(BEHAVIOR_TAGS, frozenset({"preference", "behavior"}))


if __name__ == "__main__":
    unittest.main()
