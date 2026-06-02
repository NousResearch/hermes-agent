"""Tests for scripts/code-scan/hub_ranking.py — Phase 4 D4.

Architectural Hub Ranking: compute in-degree / out-degree for project file
nodes, score hubs deterministically, optionally prefer project-local edges
when classified-imports data is supplied, exclude non-code by default, and
emit confidence/coverage notes.

Strict TDD: tests written first, implementation follows.
"""
import copy
import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure scripts/code-scan is on sys.path for sibling imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_DIR = _PROJECT_ROOT / "scripts" / "code-scan"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

FIXTURES_DIR = _PROJECT_ROOT / "tests" / "code_scan" / "fixtures" / "hub_ranking"


# ── Helpers to build graphs / classified imports ────────────────────────

def _make_graph(nodes, edges=None):
    """Build a minimal graph dict from node lists and edge lists."""
    node_dicts = []
    for nid in nodes:
        fp = nid.split(":", 1)[-1] if ":" in nid else nid
        lang = _infer_lang(fp)
        node_dicts.append({
            "node_id": nid,
            "node_type": "file",
            "filePath": fp,
            "language": lang,
        })
    return {"nodes": node_dicts, "edges": edges or []}


def _infer_lang(filepath):
    """Best-effort language inference for test fixtures."""
    if filepath.endswith(".py"):
        return "python"
    if filepath.endswith(".js") or filepath.endswith(".ts"):
        return "javascript"
    if filepath.endswith(".md"):
        return "markdown"
    if filepath.endswith(".yaml") or filepath.endswith(".yml"):
        return "yaml"
    if filepath.endswith(".json"):
        return "json"
    if filepath.endswith(".png") or filepath.endswith(".jpg"):
        return "binary"
    return "unknown"


def _make_edges(pairs):
    """Build edge dicts from iterable of (source, target) pairs."""
    return [
        {"source": s, "target": t, "edge_type": "imports"}
        for s, t in pairs
    ]


def _make_classified(file_classifications):
    """Build a classified-imports dict.

    file_classifications: dict mapping file_path -> list of (module, category)
    """
    files = {}
    for fpath, imports in file_classifications.items():
        files[fpath] = {
            "imports": [
                {"module": m, "classification": c} for m, c in imports
            ],
        }
    return {"schema_version": "1.0.0", "files": files}


# ── Dynamic import helper ───────────────────────────────────────────────

def _import_module():
    """Import hub_ranking and return key callables."""
    from hub_ranking import (
        compute_in_degree,
        compute_out_degree,
        compute_hub_scores,
        is_non_code_path,
        rank_hubs,
        _load_graph,
        _load_classified_imports,
        _build_output,
        main,
    )
    return (
        compute_in_degree,
        compute_out_degree,
        compute_hub_scores,
        is_non_code_path,
        rank_hubs,
        _load_graph,
        _load_classified_imports,
        _build_output,
        main,
    )


# ── FIXTURE: minimal graph ──────────────────────────────────────────────

@pytest.fixture
def simple_graph():
    """Graph with 4 files and import edges.

    src/app.py -> src/core.py
    src/app.py -> src/utils.py
    src/core.py -> src/utils.py
    tests/test_core.py -> src/core.py
    """
    return _make_graph(
        nodes=[
            "file:src/app.py",
            "file:src/core.py",
            "file:src/utils.py",
            "file:tests/test_core.py",
        ],
        edges=_make_edges([
            ("file:src/app.py", "file:src/core.py"),
            ("file:src/app.py", "file:src/utils.py"),
            ("file:src/core.py", "file:src/utils.py"),
            ("file:tests/test_core.py", "file:src/core.py"),
        ]),
    )


@pytest.fixture
def graph_with_docs():
    """Graph that includes non-code files."""
    return _make_graph(
        nodes=[
            "file:src/app.py",
            "file:src/core.py",
            "file:docs/README.md",
            "file:config.yaml",
            "file:assets/logo.png",
        ],
        edges=_make_edges([
            ("file:src/app.py", "file:src/core.py"),
        ]),
    )


@pytest.fixture
def graph_tie_breaking():
    """Graph where multiple nodes have identical degrees."""
    return _make_graph(
        nodes=[
            "file:src/alpha.py",
            "file:src/beta.py",
            "file:src/gamma.py",
        ],
        edges=_make_edges([
            ("file:src/alpha.py", "file:src/beta.py"),
            ("file:src/gamma.py", "file:src/beta.py"),
        ]),
    )


# ====================================================================
# RED: in-degree / out-degree computation
# ====================================================================

class TestInDegree:
    """Tests for in-degree (how many edges target each file node)."""

    def test_in_degree_leaf_node(self):
        (compute_in_degree, _, _, _, _, _, _, _, _) = _import_module()
        graph = _make_graph(
            nodes=["file:a.py", "file:b.py"],
            edges=_make_edges([("file:a.py", "file:b.py")]),
        )
        result = compute_in_degree(graph)
        assert result["file:a.py"] == 0
        assert result["file:b.py"] == 1

    def test_in_degree_multiple_incoming(self):
        (compute_in_degree, _, _, _, _, _, _, _, _) = _import_module()
        graph = _make_graph(
            nodes=["file:a.py", "file:b.py", "file:c.py"],
            edges=_make_edges([
                ("file:a.py", "file:c.py"),
                ("file:b.py", "file:c.py"),
            ]),
        )
        result = compute_in_degree(graph)
        assert result["file:c.py"] == 2

    def test_in_degree_no_edges(self):
        (compute_in_degree, _, _, _, _, _, _, _, _) = _import_module()
        graph = _make_graph(
            nodes=["file:lonely.py"],
            edges=[],
        )
        result = compute_in_degree(graph)
        assert result["file:lonely.py"] == 0

    def test_in_degree_ignores_non_file_nodes(self):
        (compute_in_degree, _, _, _, _, _, _, _, _) = _import_module()
        graph = {
            "nodes": [
                {"node_id": "file:a.py", "node_type": "file", "filePath": "a.py", "language": "python"},
                {"node_id": "module:os", "node_type": "module", "filePath": None, "language": "unknown"},
            ],
            "edges": _make_edges([("file:a.py", "module:os")]),
        }
        result = compute_in_degree(graph)
        assert "file:a.py" in result
        assert "module:os" not in result


class TestOutDegree:
    """Tests for out-degree (how many edges each file node sources)."""

    def test_out_degree_leaf_node(self):
        (_, compute_out_degree, _, _, _, _, _, _, _) = _import_module()
        graph = _make_graph(
            nodes=["file:a.py", "file:b.py"],
            edges=_make_edges([("file:a.py", "file:b.py")]),
        )
        result = compute_out_degree(graph)
        assert result["file:a.py"] == 1
        assert result["file:b.py"] == 0

    def test_out_degree_multiple_outgoing(self):
        (_, compute_out_degree, _, _, _, _, _, _, _) = _import_module()
        graph = _make_graph(
            nodes=["file:a.py", "file:b.py", "file:c.py"],
            edges=_make_edges([
                ("file:a.py", "file:b.py"),
                ("file:a.py", "file:c.py"),
            ]),
        )
        result = compute_out_degree(graph)
        assert result["file:a.py"] == 2

    def test_out_degree_no_edges(self):
        (_, compute_out_degree, _, _, _, _, _, _, _) = _import_module()
        graph = _make_graph(
            nodes=["file:lonely.py"],
            edges=[],
        )
        result = compute_out_degree(graph)
        assert result["file:lonely.py"] == 0


class TestHubScoring:
    """Tests for the hub score formula: in_degree + out_degree (default)."""

    def test_score_is_sum_of_degrees(self, simple_graph):
        (_, _, compute_hub_scores, _, _, _, _, _, _) = _import_module()
        scores = compute_hub_scores(simple_graph)
        # src/core.py: in_degree=2 (from app, test), out_degree=1 (to utils) → 3
        assert scores["file:src/core.py"] == 3.0
        # src/utils.py: in_degree=2 (from app, core), out_degree=0 → 2
        assert scores["file:src/utils.py"] == 2.0
        # src/app.py: in_degree=0, out_degree=2 → 2
        assert scores["file:src/app.py"] == 2.0

    def test_score_with_classified_imports(self, simple_graph):
        (_, _, compute_hub_scores, _, _, _, _, _, _) = _import_module()
        # Graph: a.py → b.py, a.py → c.py (both file-to-file edges)
        graph = _make_graph(
            nodes=["file:a.py", "file:b.py", "file:c.py"],
            edges=_make_edges([
                ("file:a.py", "file:b.py"),
                ("file:a.py", "file:c.py"),
            ]),
        )
        # Classified: a.py imports b (local) but c is third_party
        classified = _make_classified({
            "a.py": [
                ("b", "local"),
                ("c", "third_party"),
            ],
            "b.py": [],
            "c.py": [],
        })
        base_scores = compute_hub_scores(graph)
        classified_scores = compute_hub_scores(graph, classified=classified)
        # WITHOUT classified: a.py out=2, b.py in=1, c.py in=1
        # WITH classified: a.py→b is local ✓, a.py→c is third_party ✗
        # So c.py classified score = 0 (no local incoming) vs base = 1
        assert base_scores["file:c.py"] == 1.0
        assert classified_scores["file:c.py"] == 0.0
        # a.py has one local outgoing → score = 1 vs base = 2
        assert base_scores["file:a.py"] == 2.0
        assert classified_scores["file:a.py"] == 1.0

    def test_score_with_third_party_ignored(self):
        (_, _, compute_hub_scores, _, _, _, _, _, _) = _import_module()
        graph = _make_graph(
            nodes=["file:a.py", "file:b.py"],
            edges=_make_edges([
                ("file:a.py", "file:b.py"),
                ("file:a.py", "module:os"),
                ("file:a.py", "module:requests"),
            ]),
        )
        classified = _make_classified({
            "a.py": [
                ("b", "local"),
                ("os", "stdlib"),
                ("requests", "third_party"),
            ],
        })
        base_scores = compute_hub_scores(graph)
        classified_scores = compute_hub_scores(graph, classified=classified)
        # With classified imports, third_party/stdlib edges should not add
        # to the local-hub score, so the score should be lower or equal
        assert classified_scores["file:a.py"] <= base_scores["file:a.py"]
        # But the local edge still counts
        assert classified_scores["file:a.py"] >= 1


# ====================================================================
# RED: non-code exclusion
# ====================================================================

class TestNonCodeExclusion:
    """Tests for is_non_code_path helper."""

    def test_docs_excluded(self):
        (_, _, _, is_non_code_path, _, _, _, _, _) = _import_module()
        assert is_non_code_path("docs/README.md") is True
        assert is_non_code_path("doc/api.md") is True
        assert is_non_code_path("documentation/guide.txt") is True

    def test_config_excluded(self):
        (_, _, _, is_non_code_path, _, _, _, _, _) = _import_module()
        assert is_non_code_path("config.yaml") is True
        assert is_non_code_path("config.json") is True
        assert is_non_code_path(".env") is True
        assert is_non_code_path("pyproject.toml") is True
        assert is_non_code_path("package.json") is True
        assert is_non_code_path("Makefile") is True

    def test_assets_excluded(self):
        (_, _, _, is_non_code_path, _, _, _, _, _) = _import_module()
        assert is_non_code_path("assets/logo.png") is True
        assert is_non_code_path("static/style.css") is True
        assert is_non_code_path("images/banner.jpg") is True

    def test_code_included(self):
        (_, _, _, is_non_code_path, _, _, _, _, _) = _import_module()
        assert is_non_code_path("src/app.py") is False
        assert is_non_code_path("lib/core.js") is False
        assert is_non_code_path("main.go") is False

    def test_graphs_excluded(self):
        (_, _, _, is_non_code_path, _, _, _, _, _) = _import_module()
        assert is_non_code_path("docs/graphviz/diagram.svg") is True


# ====================================================================
# RED: ranking and filtering
# ====================================================================

class TestRankHubs:
    """Tests for the rank_hubs integration function."""

    def test_ranking_sorted_descending(self, simple_graph):
        from hub_ranking import compute_in_degree, compute_out_degree, rank_hubs
        in_deg = compute_in_degree(simple_graph)
        out_deg = compute_out_degree(simple_graph)
        scores = {n: in_deg[n] + out_deg[n] for n in in_deg}
        result = rank_hubs(simple_graph, scores, top=10)
        hub_ids = [h["node_id"] for h in result]
        # core (score 3) should come first
        assert hub_ids[0] == "file:src/core.py"

    def test_top_n_limit(self, simple_graph):
        from hub_ranking import compute_in_degree, compute_out_degree, rank_hubs
        in_deg = compute_in_degree(simple_graph)
        out_deg = compute_out_degree(simple_graph)
        scores = {n: in_deg[n] + out_deg[n] for n in in_deg}
        result = rank_hubs(simple_graph, scores, top=1)
        assert len(result) == 1

    def test_non_code_excluded_by_default(self, graph_with_docs):
        from hub_ranking import compute_in_degree, compute_out_degree, rank_hubs
        in_deg = compute_in_degree(graph_with_docs)
        out_deg = compute_out_degree(graph_with_docs)
        scores = {n: in_deg[n] + out_deg[n] for n in in_deg}
        result = rank_hubs(graph_with_docs, scores, top=10)
        hub_files = [h["file_path"] for h in result]
        assert "docs/README.md" not in hub_files
        assert "config.yaml" not in hub_files
        assert "assets/logo.png" not in hub_files

    def test_non_code_included_when_requested(self, graph_with_docs):
        from hub_ranking import compute_in_degree, compute_out_degree, rank_hubs
        in_deg = compute_in_degree(graph_with_docs)
        out_deg = compute_out_degree(graph_with_docs)
        scores = {n: in_deg[n] + out_deg[n] for n in in_deg}
        result = rank_hubs(
            graph_with_docs, scores, top=10, include_non_code=True,
        )
        hub_files = [h["file_path"] for h in result]
        # Non-code files should be present
        assert "docs/README.md" in hub_files

    def test_tie_breaking_is_deterministic(self, graph_tie_breaking):
        from hub_ranking import compute_in_degree, compute_out_degree, rank_hubs
        in_deg = compute_in_degree(graph_tie_breaking)
        out_deg = compute_out_degree(graph_tie_breaking)
        scores = {n: in_deg[n] + out_deg[n] for n in in_deg}
        # beta has in_degree=2 (from alpha, gamma), out_degree=0 → 2
        # alpha has in_degree=0, out_degree=1 → 1
        # gamma has in_degree=0, out_degree=1 → 1
        # alpha and gamma are tied at 1
        result = rank_hubs(graph_tie_breaking, scores, top=10)
        hub_ids = [h["node_id"] for h in result]
        # Tied nodes should be sorted by node_id lexicographically
        # file:src/alpha.py < file:src/gamma.py
        alpha_idx = hub_ids.index("file:src/alpha.py")
        gamma_idx = hub_ids.index("file:src/gamma.py")
        assert alpha_idx < gamma_idx, "Tie-breaking should be lexicographic by node_id"

    def test_empty_graph(self):
        (_, _, _, _, rank_hubs, _, _, _, _) = _import_module()
        graph = _make_graph(nodes=[], edges=[])
        result = rank_hubs(graph, {}, top=10)
        assert result == []

    def test_hub_entry_schema(self, simple_graph):
        from hub_ranking import compute_in_degree, compute_out_degree, rank_hubs
        in_deg = compute_in_degree(simple_graph)
        out_deg = compute_out_degree(simple_graph)
        scores = {n: in_deg[n] + out_deg[n] for n in in_deg}
        result = rank_hubs(simple_graph, scores, top=10)
        hub = result[0]
        assert "node_id" in hub
        assert "file_path" in hub
        assert "hub_score" in hub
        assert "in_degree" in hub
        assert "out_degree" in hub
        assert "confidence" in hub

    def test_no_classified_data_gives_low_confidence(self, simple_graph):
        from hub_ranking import compute_in_degree, compute_out_degree, rank_hubs
        in_deg = compute_in_degree(simple_graph)
        out_deg = compute_out_degree(simple_graph)
        scores = {n: in_deg[n] + out_deg[n] for n in in_deg}
        result = rank_hubs(simple_graph, scores, top=10, classified=None)
        for hub in result:
            assert hub["confidence"] == "low"

    def test_full_classified_coverage_gives_high_confidence(self, simple_graph):
        from hub_ranking import compute_in_degree, compute_out_degree, rank_hubs
        in_deg = compute_in_degree(simple_graph)
        out_deg = compute_out_degree(simple_graph)
        scores = {n: in_deg[n] + out_deg[n] for n in in_deg}
        classified = _make_classified({
            "src/app.py": [("src.core", "local"), ("src.utils", "local")],
            "src/core.py": [("src.utils", "local")],
            "tests/test_core.py": [("src.core", "local")],
            "src/utils.py": [],
        })
        result = rank_hubs(
            simple_graph, scores, top=10, classified=classified,
        )
        for hub in result:
            assert hub["confidence"] == "high"

    def test_partial_classified_gives_medium_confidence(self, simple_graph):
        from hub_ranking import compute_in_degree, compute_out_degree, rank_hubs
        in_deg = compute_in_degree(simple_graph)
        out_deg = compute_out_degree(simple_graph)
        scores = {n: in_deg[n] + out_deg[n] for n in in_deg}
        # Only 2 out of 4 files have classified data → partial coverage
        classified = _make_classified({
            "src/app.py": [("src.core", "local")],
            "src/core.py": [("src.utils", "local")],
            # tests/test_core.py and src/utils.py missing
        })
        result = rank_hubs(
            simple_graph, scores, top=10, classified=classified,
        )
        any_medium = any(h["confidence"] == "medium" for h in result)
        any_low = any(h["confidence"] == "low" for h in result)
        assert any_medium or any_low, (
            "Partial classification coverage should yield medium or low confidence"
        )


# ====================================================================
# RED: output building
# ====================================================================

class TestBuildOutput:
    """Tests for _build_output schema compliance."""

    def test_output_schema_version(self, simple_graph):
        from hub_ranking import compute_in_degree, compute_out_degree, rank_hubs, _build_output
        in_deg = compute_in_degree(simple_graph)
        out_deg = compute_out_degree(simple_graph)
        scores = {n: in_deg[n] + out_deg[n] for n in in_deg}
        ranked = rank_hubs(simple_graph, scores, top=20)
        output = _build_output(ranked, top=20, classification_present=False)
        assert output["schema_version"] == "1.0.0"
        assert "hub_rankings" in output
        assert "entrypoint_like" in output
        assert "totals" in output

    def test_output_totals(self, simple_graph):
        from hub_ranking import compute_in_degree, compute_out_degree, rank_hubs, _build_output
        in_deg = compute_in_degree(simple_graph)
        out_deg = compute_out_degree(simple_graph)
        scores = {n: in_deg[n] + out_deg[n] for n in in_deg}
        ranked = rank_hubs(simple_graph, scores, top=2)
        output = _build_output(ranked, top=2, classification_present=False)
        assert output["totals"]["files_ranked"] == 2
        assert output["totals"]["top_n"] == 2

    def test_output_contains_disclaimer(self, simple_graph):
        from hub_ranking import compute_in_degree, compute_out_degree, rank_hubs, _build_output
        in_deg = compute_in_degree(simple_graph)
        out_deg = compute_out_degree(simple_graph)
        scores = {n: in_deg[n] + out_deg[n] for n in in_deg}
        ranked = rank_hubs(simple_graph, scores, top=5)
        output = _build_output(ranked, top=5, classification_present=False)
        # The output should contain a disclaimer that scores are ranking hints
        disclaimer = output.get("disclaimer", "")
        assert "hint" in disclaimer.lower() or "ranking" in disclaimer.lower()


# ====================================================================
# RED: classified import integration
# ====================================================================

class TestClassifiedImportIntegration:
    """Tests for using classified imports to prefer project-local edges."""

    def test_local_edges_boost_score(self):
        from hub_ranking import compute_hub_scores
        graph = _make_graph(
            nodes=["file:src/app.py", "file:src/core.py", "file:src/utils.py"],
            edges=_make_edges([
                ("file:src/app.py", "file:src/core.py"),
                ("file:src/app.py", "file:src/utils.py"),
                ("file:src/app.py", "module:requests"),
                ("file:src/core.py", "module:os"),
            ]),
        )
        classified = _make_classified({
            "src/app.py": [
                ("src.core", "local"),
                ("src.utils", "local"),
                ("requests", "third_party"),
            ],
            "src/core.py": [
                ("os", "stdlib"),
            ],
        })
        scores = compute_hub_scores(graph, classified=classified)
        # With classified imports, only local edges boost the score
        # src/app.py: 2 local outgoing, 0 incoming local → some weighted score
        # src/core.py: 1 local incoming, 0 outgoing local → some weighted score
        assert "file:src/app.py" in scores
        assert "file:src/core.py" in scores

    def test_no_classified_uses_all_edges(self):
        from hub_ranking import compute_hub_scores
        graph = _make_graph(
            nodes=["file:src/app.py", "file:src/core.py"],
            edges=_make_edges([
                ("file:src/app.py", "file:src/core.py"),
                ("file:src/app.py", "module:os"),
            ]),
        )
        scores = compute_hub_scores(graph, classified=None)
        # Without classified, all file-to-file edges count equally.
        # The edge to module:os is excluded (non-file node).
        # app.py: in=0, out=1 -> score=1.0
        # core.py: in=1, out=0 -> score=1.0
        assert scores["file:src/app.py"] == 1.0
        assert scores["file:src/core.py"] == 1.0


# ====================================================================
# RED: CLI / main
# ====================================================================

class TestCLI:
    """Tests for the CLI entry point."""

    def test_missing_graph_file(self):
        (_, _, _, _, _, _, _, _, main_cli) = _import_module()
        test_args = ["hub_ranking.py"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main_cli()
        assert exc_info.value.code != 0

    def test_nonexistent_graph_file(self, tmp_path):
        (_, _, _, _, _, _, _, _, main_cli) = _import_module()
        graph_path = str(tmp_path / "nonexistent.json")
        test_args = ["hub_ranking.py", graph_path]
        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                rc = main_cli()
        assert rc != 0

    def test_cli_basic_output(self, simple_graph, tmp_path):
        (_, _, _, _, _, _, _, _, main_cli) = _import_module()
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(simple_graph))
        test_args = ["hub_ranking.py", str(graph_file)]
        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                rc = main_cli()
        assert rc == 0
        output = json.loads(fake_out.getvalue())
        assert output["schema_version"] == "1.0.0"
        assert len(output["hub_rankings"]) > 0

    def test_cli_top_n(self, simple_graph, tmp_path):
        (_, _, _, _, _, _, _, _, main_cli) = _import_module()
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(simple_graph))
        test_args = ["hub_ranking.py", str(graph_file), "--top", "1"]
        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                rc = main_cli()
        assert rc == 0
        output = json.loads(fake_out.getvalue())
        assert len(output["hub_rankings"]) == 1
        assert output["totals"]["top_n"] == 1

    def test_cli_with_classified_imports(self, simple_graph, tmp_path):
        (_, _, _, _, _, _, _, _, main_cli) = _import_module()
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(simple_graph))
        classified_file = tmp_path / "classified.json"
        classified = _make_classified({
            "src/app.py": [("src.core", "local")],
            "src/core.py": [("src.utils", "local")],
            "tests/test_core.py": [("src.core", "local")],
            "src/utils.py": [],
        })
        classified_file.write_text(json.dumps(classified))
        test_args = [
            "hub_ranking.py", str(graph_file),
            "--classified-imports", str(classified_file),
        ]
        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                rc = main_cli()
        assert rc == 0
        output = json.loads(fake_out.getvalue())
        assert "hub_rankings" in output
