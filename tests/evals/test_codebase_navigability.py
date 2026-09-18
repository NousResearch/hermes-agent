"""Guardrail for the navigability eval: its symbol resolver must follow the facade/sibling layout.

Two invariants the harness relies on (and that a future refactor could silently break):
  1. a name imported from a facade that lives in a sibling resolves to the SIBLING (through the facade's
     top-level `from sibling import name` or its PLUGIN-COMPAT lazy table);
  2. a name defined in the facade itself resolves to the facade.
Both are checked against real modules on the current tree, so they also pin the layout the eval documents.
"""
from pathlib import Path

import pytest

from evals.codebase_navigability import bench

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mods():
    return bench.source_modules(ROOT)


def test_resolver_follows_reexport_to_defining_sibling(mods):
    cache: dict = {}
    # gateway.run re-exports names it no longer defines; each must resolve to a gateway.run_* sibling
    src = (ROOT / "gateway" / "run.py").read_text(encoding="utf-8")
    rex = bench._reexports(src)
    moved = [(n, m) for n, (m, _) in rex.items() if m.startswith("gateway.run_")]
    assert moved, "expected gateway.run to re-export from gateway.run_* siblings"
    for name, expected in moved[:20]:
        got = bench.resolve_definer(mods, name, "gateway.run", cache)
        assert got == expected, (name, got, expected)


def test_resolver_keeps_facade_defined_names_on_facade(mods):
    cache: dict = {}
    src = (ROOT / "hermes_state.py").read_text(encoding="utf-8")
    own = [n for n, (_, _, node) in bench.top_level_defs(src).items() if not bench._is_alias(node)]
    assert own, "hermes_state.py should still define something at top level"
    for name in own[:20]:
        assert bench.resolve_definer(mods, name, "hermes_state", cache) == "hermes_state", name


def test_tokenizer_falls_back_without_crashing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_tiktoken(name, *a, **k):
        if name == "tiktoken":
            raise ImportError
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_tiktoken)
    tok = bench._tokenizer()
    assert tok("abcdefgh") == 2
