"""Integration test: full engine loop with a stub LLM.

Requires a working Jupyter kernel (``jupyter_client`` + ``ipykernel``). Skips
itself if either is missing so CI without those deps still passes.
"""
from __future__ import annotations

import importlib

import pytest

from config import RLMConfig
from corpus_loader import load_corpus
from ingestion import ingest_directory
from rlm_engine import RLMEngine


def _kernel_available() -> bool:
    for mod in ("jupyter_client", "ipykernel"):
        try:
            importlib.import_module(mod)
        except ImportError:
            return False
    return True


pytestmark = pytest.mark.skipif(
    not _kernel_available(),
    reason="jupyter_client / ipykernel not installed",
)


class StubLLM:
    model = "stub"

    def __init__(self, scripted: list[str]) -> None:
        self._scripted = iter(scripted)

    def chat(self, messages, *, system=None, temperature=0.3, max_tokens=4096):
        return next(self._scripted)


def test_engine_answers_list_query(tmp_path):
    from pathlib import Path
    fixtures = Path(__file__).parent / "fixtures" / "tiny-corpus"
    cache = tmp_path / "cache"
    ingest_directory(fixtures, cache)
    corpus = load_corpus(cache)

    scripted = [
        # Ask for a paper list via list_papers()
        "```repl\nrows = list_papers()\nnames = sorted(r['filename'] for r in rows)\nprint(names)\n```",
        # Then emit FINAL citing two of the fixtures
        "FINAL(Two papers mention Bogoliubov: [alpha.md] and [gamma.tex].)",
    ]
    cfg = RLMConfig(max_iterations=4, enable_sub_calls=False)
    spec = {"endpoint": "anthropic", "model": "dummy", "base_url": None}

    with RLMEngine(corpus, StubLLM(scripted), spec, cfg) as eng:
        result = eng.answer("Which papers mention Bogoliubov?")

    assert result["stopped_because"] == "final"
    assert "alpha.md" in result["answer"]


def test_engine_final_var_fetches_from_kernel(tmp_path):
    from pathlib import Path
    fixtures = Path(__file__).parent / "fixtures" / "tiny-corpus"
    cache = tmp_path / "cache"
    ingest_directory(fixtures, cache)
    corpus = load_corpus(cache)

    scripted = [
        "```repl\nmy_answer = 'hello from the kernel'\nprint('ok')\n```",
        "FINAL_VAR(my_answer)",
    ]
    cfg = RLMConfig(max_iterations=4, enable_sub_calls=False)
    spec = {"endpoint": "anthropic", "model": "dummy", "base_url": None}

    with RLMEngine(corpus, StubLLM(scripted), spec, cfg) as eng:
        result = eng.answer("trivial")

    assert result["final_kind"] == "var"
    assert result["answer"] == "hello from the kernel"


def test_engine_max_iterations_exits_gracefully(tmp_path):
    from pathlib import Path
    fixtures = Path(__file__).parent / "fixtures" / "tiny-corpus"
    cache = tmp_path / "cache"
    ingest_directory(fixtures, cache)
    corpus = load_corpus(cache)

    # Model never emits FINAL or a ```repl block
    scripted = ["thinking...", "more thinking...", "still going..."]
    cfg = RLMConfig(max_iterations=3, enable_sub_calls=False)
    spec = {"endpoint": "anthropic", "model": "dummy", "base_url": None}

    with RLMEngine(corpus, StubLLM(scripted), spec, cfg) as eng:
        result = eng.answer("q")

    assert result["stopped_because"] == "max_iterations"
    assert result["answer"] is None
