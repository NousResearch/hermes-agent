"""The import-graph rules (LAYER_EDGE, IMPORT_CYCLE) and the base graph they ratchet against.

The unit cases build graphs from source strings; the git cases prove the plumbing the CLI uses
(``gitio.read_many`` for a revision, untracked files for a working tree) and that the derived
base graph is the graph a second whole-tree scan would build.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.code_health import gitio, py_imports
from scripts.code_health.cli import _graph
from scripts.code_health.config import graph_scope

_EMPTY = ""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True, encoding="utf-8", timeout=60).stdout.strip()


def _commit(repo: Path, files: dict[str, str]) -> str:
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "step")
    return _git(repo, "rev-parse", "HEAD")


def _repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    return repo, _commit(repo, files)


def _rules(findings) -> list[str]:
    return [f.rule for f in findings]


# --- unit: what counts as an edge -----------------------------------------------------------

def test_call_time_import_is_not_an_edge():
    graph = py_imports.scan({
        "agent/run.py": "def go():\n    from hermes_cli import cfg\n    return cfg\n",
        "hermes_cli/cfg.py": _EMPTY,
    })
    assert graph.edges == {}


def test_from_package_import_submodule_resolves_to_the_module():
    graph = py_imports.scan({
        "tools/loop.py": "from hermes_cli import cfg\n",
        "hermes_cli/cfg.py": _EMPTY,
        "hermes_cli/__init__.py": _EMPTY,
    })
    # both readings of `from pkg import submodule` are kept: the package and the module
    assert set(graph.edges["tools.loop"]) == {"hermes_cli", "hermes_cli.cfg"}


def test_relative_import_resolves_against_the_package():
    graph = py_imports.scan({
        "tools/loop.py": "from . import helper\n",
        "tools/helper.py": _EMPTY,
    })
    assert list(graph.edges["tools.loop"]) == ["tools.helper"]


def test_undefined_direction_is_counted_but_not_judged():
    # `gateway -> cli` is not in #14182 either way: it must not become a finding.
    graph = py_imports.scan({
        "gateway/session.py": "from hermes_cli import cfg\n",
        "hermes_cli/cfg.py": _EMPTY,
    })
    assert py_imports.gate(None, graph) == []


def test_scope_is_product_python_only():
    assert graph_scope("agent/run.py")
    assert not graph_scope("tests/scripts/test_x.py")
    assert not graph_scope("plugins/platforms/signal/__init__.py")
    assert not graph_scope("scripts/code_health/cli.py")


# --- unit: the ratchet ----------------------------------------------------------------------

def _head_with_extra_edge() -> py_imports.Graph:
    return py_imports.scan({
        "agent/run.py": "from hermes_cli import cfg\n",
        "hermes_cli/cfg.py": _EMPTY,
    })


def test_new_layer_edge_fails_with_the_line_of_the_import():
    findings = py_imports.gate(None, _head_with_extra_edge())
    assert _rules(findings) == ["LAYER_EDGE"]
    assert findings[0].path == "agent/run.py"
    assert findings[0].line == 1
    assert "cap 0" in findings[0].detail


def test_the_same_debt_frozen_on_the_base_passes():
    base = _head_with_extra_edge()
    assert py_imports.gate(base, base) == []


def test_one_more_edge_over_the_cap_fails():
    base = py_imports.scan({
        "agent/run.py": "from hermes_cli import cfg\n",
        "hermes_cli/cfg.py": _EMPTY,
        "hermes_cli/other.py": _EMPTY,
    })
    head = py_imports.scan({
        "agent/run.py": "from hermes_cli import cfg\nfrom hermes_cli import other\n",
        "hermes_cli/cfg.py": _EMPTY,
        "hermes_cli/other.py": _EMPTY,
    })
    findings = py_imports.gate(base, head)
    assert _rules(findings) == ["LAYER_EDGE"]
    assert "cap 1 on main (+1 new)" in findings[0].detail


def test_new_cycle_fails_and_a_frozen_one_passes():
    base = py_imports.scan({"tools/a.py": _EMPTY, "tools/b.py": _EMPTY})
    head = py_imports.scan({
        "tools/a.py": "from tools import b\n",
        "tools/b.py": "from tools import a\n",
    })
    findings = py_imports.gate(base, head)
    assert _rules(findings) == ["IMPORT_CYCLE"]
    assert findings[0].path in ("tools/a.py", "tools/b.py")
    assert py_imports.gate(head, head) == []


def test_a_cycle_inside_one_package_is_still_a_cycle():
    graph = py_imports.scan({
        "tools/mcp_tool_loop.py": "from tools import mcp_tool_lifecycle\n",
        "tools/mcp_tool_lifecycle.py": "from tools import mcp_tool_loop\n",
    })
    assert py_imports.sccs(graph) == [("tools.mcp_tool_lifecycle", "tools.mcp_tool_loop")]


# --- git: the plumbing the CLI runs ---------------------------------------------------------

def test_derived_base_equals_a_full_scan_of_the_base_tree(tmp_path):
    repo, base = _repo(tmp_path, {
        "agent/run.py": _EMPTY,
        "hermes_cli/cfg.py": _EMPTY,
        "tools/loop.py": "from tools import helper\n",
        "tools/helper.py": _EMPTY,
    })
    head_sha = _commit(repo, {
        "agent/run.py": "from hermes_cli import cfg\n",   # a new forbidden edge
        "tools/helper.py": "from tools import loop\n",    # closes a cycle with tools/loop.py
    })
    changes = gitio.changed_files(repo, base, head_sha)
    changed_at_head = [c.new for c in changes if c.new and graph_scope(c.new)]
    base_contents = gitio.read_many(repo, base, [c.old for c in changes if c.old and graph_scope(c.old)])
    head = _graph(repo, head_sha)
    derived = py_imports.derive_base(head, changed_at_head, base_contents)
    full = _graph(repo, base)
    assert (len(derived.modules), sum(map(len, derived.edges.values())), py_imports.sccs(derived)) == (
        len(full.modules), sum(map(len, full.edges.values())), py_imports.sccs(full))
    assert _rules(py_imports.gate(derived, head)) == ["LAYER_EDGE", "IMPORT_CYCLE"]


def test_working_tree_head_reads_untracked_files(tmp_path):
    repo, _base = _repo(tmp_path, {"agent/run.py": _EMPTY})
    (repo / "hermes_cli").mkdir()
    (repo / "hermes_cli" / "cfg.py").write_text(_EMPTY, encoding="utf-8")
    (repo / "agent" / "run.py").write_text("from hermes_cli import cfg\n", encoding="utf-8")
    graph = _graph(repo, None)
    assert "hermes_cli.cfg" in graph.modules
    assert list(graph.edges["agent.run"]) == ["hermes_cli.cfg"]


def test_one_import_statement_is_one_finding():
    # `from hermes_cli import cfg` reads as two edges (the package and the module): the report
    # names the statement once, not twice.
    findings = py_imports.gate(None, py_imports.scan({
        "agent/run.py": "from hermes_cli import cfg\n",
        "hermes_cli/cfg.py": _EMPTY,
        "hermes_cli/__init__.py": _EMPTY,
    }))
    assert _rules(findings) == ["LAYER_EDGE"]
    assert findings[0].line == 1
