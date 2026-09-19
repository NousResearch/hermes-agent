"""Graphify knowledge graph: implementation-first ranking + capped one-line hits."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "graphify_knowledge.py"


def _load():
    spec = importlib.util.spec_from_file_location("graphify_knowledge", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_symbol_search_prefers_implementation_over_tests():
    mod = _load()
    symbols = [
        {"n": "aiagent", "raw": "C:_AIAgent:11", "p": "tests/agent/test_x.py"},
        {"n": "aiagent", "raw": "C:AIAgent:229", "p": "run_agent.py"},
    ]
    hits = mod.search_symbols(symbols, "AIAgent", 5)
    assert len(hits) == 2
    # Behaviour contract: the defining module outranks its test mirror on ties.
    assert hits[0].startswith("run_agent.py")
    assert hits[1].startswith("tests/")


def test_query_hits_are_capped_single_lines():
    mod = _load()
    long_summary = "S " * 500
    files = [
        {"p": "agent/memory_provider.py", "l": 99999,
         "s": long_summary, "d": ["C:" + "N" * 300 + ":1"], "i": []},
    ]
    symbols = [{"n": "n" * 300, "raw": "C:" + "N" * 300 + ":1",
                "p": "agent/memory_provider.py"}]
    for hit in mod.search_symbols(symbols, "n" * 10, 5) + mod.search_text(files, "memory", 5):
        # Token contract: one line, hard-capped — hits never dump file contents.
        assert "\n" not in hit
        assert len(hit) <= 220
