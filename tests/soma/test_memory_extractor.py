"""Tests for soma.memory_extractor — JSON parse, validation, store integration."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from soma.memory_extractor import MemoryExtractor, _coerce_json_array
from soma.memory_store import MemoryStore


class _DictEmbedder:
    def __init__(self):
        self._slots: dict[str, int] = {}

    def embed(self, text: str) -> list[float]:
        if text not in self._slots:
            self._slots[text] = len(self._slots)
        vec = [0.0] * 64
        vec[self._slots[text] % 64] = 1.0
        return vec


def _fake_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path, _DictEmbedder())


class CoerceJsonArrayTest(unittest.TestCase):
    def test_pure_json(self):
        self.assertEqual(_coerce_json_array('[{"a": 1}]'), [{"a": 1}])

    def test_fenced_json(self):
        raw = "```json\n[{\"a\": 1}, {\"b\": 2}]\n```"
        self.assertEqual(_coerce_json_array(raw), [{"a": 1}, {"b": 2}])

    def test_preamble_then_array(self):
        raw = "Sure, here you go:\n[{\"a\": 1}]"
        self.assertEqual(_coerce_json_array(raw), [{"a": 1}])

    def test_empty_array(self):
        self.assertEqual(_coerce_json_array("[]"), [])

    def test_garbage_returns_none(self):
        self.assertIsNone(_coerce_json_array("not json at all"))
        self.assertIsNone(_coerce_json_array(""))


class MemoryExtractorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self.tmp.close()
        self.addCleanup(lambda: os.path.exists(self.tmp.name) and os.unlink(self.tmp.name))
        self.store = _store(self.tmp.name)

    def _extractor(self, llm_response: str) -> MemoryExtractor:
        call_llm = MagicMock(return_value=_fake_response(llm_response))
        return MemoryExtractor(self.store, call_llm=call_llm)

    def test_extracts_valid_memories(self):
        raw = """[
            {"type": "semantic", "content": "User works at Supermicro", "tags": ["domain", "identity"]},
            {"type": "procedural", "content": "Always confirm before rm -rf", "tags": ["behavior"]}
        ]"""
        extractor = self._extractor(raw)
        written = extractor.extract("Ich arbeite bei Supermicro", "Got it.")
        self.assertEqual(len(written), 2)
        contents = [r.content for r in written]
        self.assertIn("User works at Supermicro", contents)
        self.assertIn("Always confirm before rm -rf", contents)

    def test_invalid_type_skipped(self):
        raw = '[{"type": "magic", "content": "x", "tags": []}]'
        written = self._extractor(raw).extract("u", "a")
        self.assertEqual(len(written), 0)

    def test_empty_content_skipped(self):
        raw = '[{"type": "semantic", "content": "   ", "tags": []}]'
        written = self._extractor(raw).extract("u", "a")
        self.assertEqual(len(written), 0)

    def test_malformed_json_returns_empty(self):
        written = self._extractor("not json").extract("u", "a")
        self.assertEqual(len(written), 0)

    def test_empty_array_returns_empty(self):
        written = self._extractor("[]").extract("u", "a")
        self.assertEqual(len(written), 0)

    def test_dedup_via_store_fuse(self):
        raw = '[{"type": "semantic", "content": "User works at Supermicro", "tags": ["domain"]}]'
        extractor = self._extractor(raw)
        first = extractor.extract("Ich arbeite bei Supermicro", "ok")
        second = extractor.extract("Ich arbeite bei Supermicro", "ok")
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0].id, second[0].id)
        self.assertEqual(second[0].use_count, 2)
        self.assertEqual(len(self.store.all()), 1)

    def test_llm_exception_returns_empty(self):
        call_llm = MagicMock(side_effect=RuntimeError("provider down"))
        extractor = MemoryExtractor(self.store, call_llm=call_llm)
        written = extractor.extract("u", "a")
        self.assertEqual(len(written), 0)

    def test_empty_inputs_skip_llm_call(self):
        call_llm = MagicMock()
        extractor = MemoryExtractor(self.store, call_llm=call_llm)
        written = extractor.extract("", "")
        self.assertEqual(len(written), 0)
        call_llm.assert_not_called()

    def test_max_per_turn_caps_writes(self):
        items = ",".join(
            f'{{"type": "semantic", "content": "fact {i}", "tags": []}}'
            for i in range(20)
        )
        extractor = self._extractor(f"[{items}]")
        extractor.max_per_turn = 5
        written = extractor.extract("u", "a")
        self.assertEqual(len(written), 5)

    def test_content_truncated(self):
        long_content = "x" * 1000
        raw = f'[{{"type": "semantic", "content": "{long_content}", "tags": []}}]'
        extractor = self._extractor(raw)
        extractor.max_content_chars = 100
        written = extractor.extract("u", "a")
        self.assertEqual(len(written), 1)
        self.assertEqual(len(written[0].content), 100)

    def test_async_wrapper(self):
        raw = '[{"type": "semantic", "content": "X", "tags": []}]'
        extractor = self._extractor(raw)
        result = asyncio.run(extractor.extract_async("u", "a"))
        self.assertEqual(len(result), 1)

    def test_passes_model_override(self):
        call_llm = MagicMock(return_value=_fake_response("[]"))
        extractor = MemoryExtractor(self.store, call_llm=call_llm, model="claude-3-opus")
        extractor.extract("u", "a")
        kwargs = call_llm.call_args.kwargs
        self.assertEqual(kwargs.get("model"), "claude-3-opus")
        self.assertEqual(kwargs.get("temperature"), 0.0)
        self.assertEqual(len(kwargs.get("messages")), 2)
        self.assertEqual(kwargs["messages"][0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
