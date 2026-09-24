"""Tests for scripts/code-scan/analyze_graph.py — UA-003 Deterministic Graph Analytics Layer.

TDD cycle:
  RED:  tests fail before implementation (module missing or empty).
  GREEN: minimal implementation passes tests.
  FULL:  run all tests/code_scan tests + smoke verification.
"""
import json
import subprocess
import sys
from pathlib import Path
from io import StringIO
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ANALYZE_SCRIPT = PROJECT_ROOT / "scripts" / "code-scan" / "analyze_graph.py"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "code_scan" / "fixtures" / "graph-analytics"


def _import_analyze_graph():
    """Import analyze_graph module with sys.path pointing to scripts/code-scan."""
    script_dir = str(PROJECT_ROOT / "scripts" / "code-scan")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    # Import only what we need — the module must exist for tests to run
    import analyze_graph
    return analyze_graph


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ag():
    return _import_analyze_graph()


@pytest.fixture
def sample_graph():
    return json.loads((FIXTURES_DIR / "sample_graph.json").read_text())


@pytest.fixture
def cyclic_graph():
    return json.loads((FIXTURES_DIR / "cyclic_graph.json").read_text())


@pytest.fixture
def empty_graph():
    return json.loads((FIXTURES_DIR / "empty_graph.json").read_text())


# ── compute_degree ──────────────────────────────────────────────────────────

class TestComputeDegree:
    def test_in_degree_correct(self, ag, sample_graph):
        """In-degree of a module counts how many edges point to it."""
        in_deg = ag.compute_in_degree(sample_graph)
        # module:os has edges from src/main.py, src/utils.py, tests/test_main.py → in-degree 3
        assert in_deg.get("module:os", 0) == 3
        # module:typing has edges from src/utils.py, src/models.py → in-degree 2
        assert in_deg.get("module:typing", 0) == 2
        # file nodes have 0 in-degree in this graph (no edges target them)
        assert in_deg.get("file:src/main.py", 0) == 0

    def test_out_degree_correct(self, ag, sample_graph):
        """Out-degree of a file counts how many edges originate from it."""
        out_deg = ag.compute_out_degree(sample_graph)
        # src/main.py imports 4 modules → out-degree 4
        assert out_deg.get("file:src/main.py", 0) == 4
        # src/utils.py imports 3 modules → out-degree 3
        assert out_deg.get("file:src/utils.py", 0) == 3
        # modules have 0 out-degree in this graph
        assert out_deg.get("module:os", 0) == 0

    def test_empty_graph_degrees(self, ag, empty_graph):
        in_deg = ag.compute_in_degree(empty_graph)
        out_deg = ag.compute_out_degree(empty_graph)
        assert in_deg == {}
        assert out_deg == {}


# ── top_k_by_degree ────────────────────────────────────────────────────────

class TestTopKByDegree:
    def test_top_in_degree_hub(self, ag, sample_graph):
        in_deg = ag.compute_in_degree(sample_graph)
        top = ag.top_k_by_degree(in_deg, sample_graph, k=3, degree_type="in")
        assert len(top) <= 3
        # module:os should be first (in-degree 3)
        assert top[0]["node_id"] == "module:os"
        assert top[0]["degree"] == 3

    def test_top_out_degree_hub(self, ag, sample_graph):
        out_deg = ag.compute_out_degree(sample_graph)
        top = ag.top_k_by_degree(out_deg, sample_graph, k=3, degree_type="out")
        assert len(top) <= 3
        # file:src/main.py should be first (out-degree 4)
        assert top[0]["node_id"] == "file:src/main.py"
        assert top[0]["degree"] == 4

    def test_top_k_respects_limit(self, ag, sample_graph):
        out_deg = ag.compute_out_degree(sample_graph)
        top = ag.top_k_by_degree(out_deg, sample_graph, k=2, degree_type="out")
        assert len(top) == 2

    def test_empty_graph_top_k(self, ag, empty_graph):
        top = ag.top_k_by_degree({}, empty_graph, k=5, degree_type="in")
        assert top == []


# ── find_bidirectional_imports ─────────────────────────────────────────────

class TestFindBidirectionalImports:
    def test_detects_bidirectional_pair(self, ag, cyclic_graph):
        bidi = ag.find_bidirectional_imports(cyclic_graph)
        # a.py → module:b and b.py → module:a forms a bidirectional pair
        assert len(bidi) >= 1
        pair_node_ids = {tuple(sorted(p)) for p in bidi}
        assert ("file:a.py", "file:b.py") in pair_node_ids

    def test_no_bidirectional_in_sample(self, ag, sample_graph):
        bidi = ag.find_bidirectional_imports(sample_graph)
        # sample_graph has no bidirectional imports between file nodes
        assert bidi == []

    def test_empty_graph_no_bidi(self, ag, empty_graph):
        bidi = ag.find_bidirectional_imports(empty_graph)
        assert bidi == []


# ── find_entrypoint_candidates ─────────────────────────────────────────────

class TestFindEntrypointCandidates:
    def test_main_file_detected(self, ag, sample_graph):
        candidates = ag.find_entrypoint_candidates(sample_graph)
        ids = [c["node_id"] for c in candidates]
        # src/main.py should be detected as entrypoint (has "main" in name/path)
        assert any("main" in nid for nid in ids)

    def test_entrypoint_has_confidence(self, ag, sample_graph):
        candidates = ag.find_entrypoint_candidates(sample_graph)
        for c in candidates:
            assert "confidence" in c
            assert "reason" in c
            assert 0 <= c["confidence"] <= 1

    def test_empty_graph_no_entrypoints(self, ag, empty_graph):
        candidates = ag.find_entrypoint_candidates(empty_graph)
        assert candidates == []


# ── directory_summary ──────────────────────────────────────────────────────

class TestDirectorySummary:
    def test_directory_groups(self, ag, sample_graph):
        summary = ag.compute_directory_summary(sample_graph)
        assert isinstance(summary, dict)
        # Should have at least "src" and "tests" directories
        dirs_with_files = [d for d in summary if summary[d].get("file_count", 0) > 0]
        assert "src" in dirs_with_files
        assert "tests" in dirs_with_files

    def test_directory_has_edge_counts(self, ag, sample_graph):
        summary = ag.compute_directory_summary(sample_graph)
        for dir_name, dir_data in summary.items():
            assert "file_count" in dir_data
            assert "internal_edges" in dir_data
            assert "external_edges" in dir_data

    def test_empty_graph_directory_summary(self, ag, empty_graph):
        summary = ag.compute_directory_summary(empty_graph)
        assert summary == {}


# ── review_priority_candidates ─────────────────────────────────────────────

class TestReviewPriorityCandidates:
    def test_high_out_degree_is_priority(self, ag, sample_graph):
        priorities = ag.find_review_priority_candidates(sample_graph)
        assert isinstance(priorities, list)
        # src/main.py has highest out-degree, should be a candidate
        ids = [c["node_id"] for c in priorities]
        assert any("src/main.py" in nid for nid in ids)

    def test_priority_has_reason(self, ag, sample_graph):
        priorities = ag.find_review_priority_candidates(sample_graph)
        for p in priorities:
            assert "reason" in p
            assert "score" in p

    def test_empty_graph_no_priorities(self, ag, empty_graph):
        priorities = ag.find_review_priority_candidates(empty_graph)
        assert priorities == []


# ── analyze_graph full pipeline ────────────────────────────────────────────

class TestAnalyzeGraph:
    def test_full_analysis_output_shape(self, ag, sample_graph):
        result = ag.analyze_graph(sample_graph)
        assert isinstance(result, dict)
        # Required top-level keys per bead specification
        assert any(k in result for k in [
            "top_in_degree", "top_out_degree", "hubs", "entrypoint_candidates"
        ])
        assert "directory_summary" in result
        assert "review_priority" in result
        assert "bidirectional_imports" in result

    def test_deterministic_output(self, ag, sample_graph):
        """Two runs on the same input must produce identical output."""
        r1 = ag.analyze_graph(sample_graph)
        r2 = ag.analyze_graph(sample_graph)
        # Serialize for comparison (excluding any timestamp if present)
        def strip_ts(d):
            return {k: v for k, v in d.items() if k != "generated_at"}
        assert strip_ts(r1) == strip_ts(r2)

    def test_output_labels_hints_not_truth(self, ag, sample_graph):
        """Analytics are hints, not truth. Output must label confidence."""
        result = ag.analyze_graph(sample_graph)
        # entrypoint_candidates must have confidence labels
        if result.get("entrypoint_candidates"):
            for c in result["entrypoint_candidates"]:
                assert "confidence" in c

    def test_empty_graph_analysis(self, ag, empty_graph):
        result = ag.analyze_graph(empty_graph)
        assert isinstance(result, dict)
        # Should not error, just produce empty collections
        assert result.get("top_in_degree", []) == []
        assert result.get("top_out_degree", []) == []


# ── CLI entry point ────────────────────────────────────────────────────────

class TestCLI:
    def test_cli_outputs_valid_json(self):
        """CLI must emit valid JSON to stdout."""
        fixture = str(FIXTURES_DIR / "sample_graph.json")
        result = subprocess.run(
            [sys.executable, str(ANALYZE_SCRIPT), fixture],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert isinstance(data, dict)

    def test_cli_has_required_keys(self):
        """CLI output must contain at least one required key."""
        fixture = str(FIXTURES_DIR / "sample_graph.json")
        result = subprocess.run(
            [sys.executable, str(ANALYZE_SCRIPT), fixture],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        assert any(k in data for k in [
            "top_in_degree", "top_out_degree", "hubs", "entrypoint_candidates"
        ])

    def test_cli_no_log_on_stdout(self):
        """Progress/log messages must go to stderr, not stdout."""
        fixture = str(FIXTURES_DIR / "sample_graph.json")
        result = subprocess.run(
            [sys.executable, str(ANALYZE_SCRIPT), fixture],
            capture_output=True, text=True, timeout=30,
        )
        # stdout must be parseable JSON with no extra text
        json.loads(result.stdout)  # will raise if stdout has log lines

    def test_cli_missing_file(self):
        """CLI must return non-zero for missing input file."""
        result = subprocess.run(
            [sys.executable, str(ANALYZE_SCRIPT), "/nonexistent/graph.json"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0

    def test_cli_cyclic_graph(self):
        """CLI must handle cyclic graph correctly."""
        fixture = str(FIXTURES_DIR / "cyclic_graph.json")
        result = subprocess.run(
            [sys.executable, str(ANALYZE_SCRIPT), fixture],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert isinstance(data, dict)

    def test_cli_output_path(self):
        """CLI --output flag writes to file."""
        import tempfile
        fixture = str(FIXTURES_DIR / "sample_graph.json")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                [sys.executable, str(ANALYZE_SCRIPT), fixture, "--output", tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            data = json.loads(Path(tmp_path).read_text())
            assert isinstance(data, dict)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
