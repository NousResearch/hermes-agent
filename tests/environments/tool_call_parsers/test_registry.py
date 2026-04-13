"""Registry-level integrity tests.

Asserts:
  - every registered parser has a dedicated test file (or shares one
    via inheritance), so adding a new parser without tests fails CI;
  - get_parser raises a clear KeyError on unknown names;
  - all parsers honour the no-tool-call passthrough contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from environments.tool_call_parsers import get_parser, list_parsers


# Every parser name → the test file that exercises it.
EXPECTED_TEST_FILES = {
    "hermes": "test_hermes.py",
    "longcat": "test_longcat.py",
    "qwen": "test_qwen.py",
    "mistral": "test_mistral.py",
    "llama3_json": "test_llama.py",
    "llama4_json": "test_llama.py",
    "deepseek_v3": "test_deepseek_v3.py",
    "deepseek_v3_1": "test_deepseek_v3_1.py",
    "deepseek_v31": "test_deepseek_v3_1.py",
    "kimi_k2": "test_kimi_k2.py",
    "glm45": "test_glm45.py",
    "glm47": "test_glm47.py",
    "qwen3_coder": "test_qwen3_coder.py",
}


class TestRegistry:
    def test_unknown_parser_raises(self):
        with pytest.raises(KeyError) as excinfo:
            get_parser("does_not_exist")
        assert "Available parsers" in str(excinfo.value)

    def test_every_registered_parser_has_a_test_file(self):
        """If you add a new parser, also add tests for it."""
        names = set(list_parsers())
        documented = set(EXPECTED_TEST_FILES.keys())
        missing = names - documented
        assert not missing, (
            f"Parsers without coverage entries in EXPECTED_TEST_FILES: {sorted(missing)}"
        )

    def test_every_documented_test_file_actually_exists(self):
        """Catches docstring drift: EXPECTED_TEST_FILES claims a test file
        that no one actually wrote."""
        here = Path(__file__).parent
        missing = []
        for name, file_ in EXPECTED_TEST_FILES.items():
            if not (here / file_).is_file():
                missing.append((name, file_))
        assert not missing, f"missing test files: {missing}"


class TestPassthroughContract:
    """Every parser must return (raw_text, None) when there is no
    tool call markup at all. This is the universal cheap-path invariant."""

    @pytest.mark.parametrize("name", sorted(EXPECTED_TEST_FILES.keys()))
    def test_plain_text_passes_through_unchanged(self, name):
        parser = get_parser(name)
        plain = "Just some plain conversational text."
        content, tool_calls = parser.parse(plain)
        assert tool_calls is None, f"{name} produced spurious tool_calls on plain text"
        assert content == plain, f"{name} mutated plain text"
