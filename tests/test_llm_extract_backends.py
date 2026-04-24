"""Tests for LLM extract backend selection (codex / anthropic / auto).

Validates the multi-backend flow in MemoryExtractionMiddleware._llm_extract
without calling real services.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from agent_bus.middlewares.memory_extraction import (
    MemoryExtractionMiddleware,
    _build_extract_prompt,
    _parse_facts,
)


class TestPromptBuilder:
    def test_prompt_contains_instructions_and_convo(self):
        p = _build_extract_prompt(["user: hi", "assistant: hello"])
        assert "JSON array" in p
        assert "preference" in p  # category vocab
        assert "user: hi" in p
        assert "---" in p


class TestParseFacts:
    def test_well_formed_array(self):
        raw = '[{"content":"X likes coffee","category":"preference","confidence":0.9}]'
        facts = _parse_facts(raw, source_tag="codex:gpt", thread_id="t1")
        assert len(facts) == 1
        assert facts[0].content == "X likes coffee"
        assert facts[0].category == "preference"
        assert facts[0].confidence == 0.9
        assert facts[0].source == "codex:gpt"

    def test_wrapped_in_prose(self):
        raw = 'Here you go:\n```json\n[{"content":"A", "category":"knowledge"}]\n```\nDone.'
        facts = _parse_facts(raw, source_tag="codex:gpt", thread_id="t1")
        assert len(facts) == 1
        assert facts[0].content == "A"

    def test_unknown_category_falls_back(self):
        raw = '[{"content":"X","category":"weird","confidence":0.5}]'
        facts = _parse_facts(raw, source_tag="x", thread_id="t")
        assert facts[0].category == "context"

    def test_confidence_clamped(self):
        raw = '[{"content":"X","category":"preference","confidence":9.0}]'
        facts = _parse_facts(raw, source_tag="x", thread_id="t")
        assert facts[0].confidence == 1.0

    def test_too_long_content_dropped(self):
        long = "A" * 201
        raw = f'[{{"content":"{long}","category":"preference"}}]'
        facts = _parse_facts(raw, source_tag="x", thread_id="t")
        assert facts == []

    def test_no_array_in_text(self):
        facts = _parse_facts("sorry, no JSON", source_tag="x", thread_id="t")
        assert facts == []

    def test_invalid_json(self):
        facts = _parse_facts("[oops not json]", source_tag="x", thread_id="t")
        assert facts == []


class TestBackendSelection:
    def test_no_backends_no_key_no_cli_returns_empty(self):
        mw = MemoryExtractionMiddleware()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ["HERMES_AUTO_MEMORY_BACKEND"] = "anthropic"  # skip codex
            facts = mw._llm_extract("t", [{"role": "user", "content": "I prefer X"}])
            assert facts == []

    def test_codex_backend_subprocess_mocked_success(self):
        """Verify that a successful codex subprocess call produces facts."""
        mw = MemoryExtractionMiddleware()
        fake_stdout = '[{"content":"User prefers black coffee","category":"preference","confidence":0.8}]'

        class FakeProc:
            returncode = 0
            stdout = fake_stdout
            stderr = ""

        with mock.patch.dict(os.environ, {"HERMES_AUTO_MEMORY_BACKEND": "codex"}):
            with mock.patch("shutil.which", return_value="/fake/codex"):
                with mock.patch("subprocess.run", return_value=FakeProc()):
                    facts = mw._llm_extract("t", [{"role": "user", "content": "I drink black coffee"}])
        assert len(facts) == 1
        assert "coffee" in facts[0].content
        assert facts[0].source.startswith("codex:")

    def test_codex_missing_falls_through_in_auto_mode(self):
        """auto mode tries codex first; if codex CLI missing, try anthropic."""
        mw = MemoryExtractionMiddleware()
        with mock.patch.dict(os.environ, {"HERMES_AUTO_MEMORY_BACKEND": "auto"}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with mock.patch("shutil.which", return_value=None):  # no codex
                facts = mw._llm_extract("t", [{"role": "user", "content": "hi"}])
        assert facts == []  # both backends unavailable

    def test_codex_timeout_returns_empty(self):
        """Timeout in codex should be treated as backend-unavailable."""
        import subprocess as _sp
        mw = MemoryExtractionMiddleware()
        with mock.patch.dict(os.environ, {"HERMES_AUTO_MEMORY_BACKEND": "codex"}):
            with mock.patch("shutil.which", return_value="/fake/codex"):
                with mock.patch("subprocess.run", side_effect=_sp.TimeoutExpired(cmd="codex", timeout=1)):
                    facts = mw._llm_extract("t", [{"role": "user", "content": "hi"}])
        assert facts == []

    def test_codex_nonzero_exit_returns_empty(self):
        mw = MemoryExtractionMiddleware()

        class FakeProc:
            returncode = 1
            stdout = ""
            stderr = "error"

        with mock.patch.dict(os.environ, {"HERMES_AUTO_MEMORY_BACKEND": "codex"}):
            with mock.patch("shutil.which", return_value="/fake/codex"):
                with mock.patch("subprocess.run", return_value=FakeProc()):
                    facts = mw._llm_extract("t", [{"role": "user", "content": "hi"}])
        assert facts == []
