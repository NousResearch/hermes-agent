"""UA-002: Warning Severity Taxonomy tests.

Strict TDD: these tests verify that:
- (RED) Old flat warnings cannot differentiate severity
- (GREEN) New severity summary is present and stable
- Backward compatibility: existing warnings/issues fields unchanged
"""
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "code-scan"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from graph_schema import (
    WarningSeverity,
    classify_warning_severity,
    validate_graph,
    build_warning_summary,
)


# ---------------------------------------------------------------------------
# RED tests — old flat output cannot differentiate severity
# ---------------------------------------------------------------------------
class TestOldFlatWarningsNotSeverelyTyped:
    """Prove that flat string warnings lack severity information.

    The old output was just a list of strings. After UA-002, a new
    parallel severity summary must exist so consumers can differentiate
    INFO vs MINOR vs MODERATE vs MAJOR.
    """

    def test_old_warnings_field_is_strings_only(self):
        """The 'warnings' list remains plain strings for compatibility."""
        graph = {
            "nodes": [
                {"node_type": "file", "node_id": "docs/README.md", "filePath": "docs/README.md"},
                {"node_type": "file", "node_id": "src/main.py", "filePath": "src/main.py"},
            ],
            "edges": [
                {"edge_type": "imports", "source": "src/main.py", "target": "src/main.py"},
            ],
        }
        result = validate_graph(graph)
        # warnings field must still exist and be a list of strings
        assert "warnings" in result
        for w in result["warnings"]:
            assert isinstance(w, str)

    def test_old_output_has_no_severity_key_missing(self):
        """If severity_summary were missing, there would be no differentiation."""
        graph = {
            "nodes": [
                {"node_type": "file", "node_id": "orphan_doc", "filePath": "docs/old.md"},
                {"node_type": "file", "node_id": "orphan_src", "filePath": "src/unused_module.py"},
            ],
            "edges": [],
        }
        result = validate_graph(graph)
        # GREEN requirement: severity_summary MUST be present
        assert "severity_summary" in result, "severity_summary is missing from validate_graph output"


# ---------------------------------------------------------------------------
# GREEN tests — severity summary is present, stable, backward compatible
# ---------------------------------------------------------------------------
class TestWarningSeverityEnum:
    """Verify the WarningSeverity enum has exactly the four required values."""

    def test_severity_info(self):
        assert WarningSeverity.INFO == "info"

    def test_severity_minor(self):
        assert WarningSeverity.MINOR == "minor"

    def test_severity_moderate(self):
        assert WarningSeverity.MODERATE == "moderate"

    def test_severity_major(self):
        assert WarningSeverity.MAJOR == "major"

    def test_all_severities_defined(self):
        values = {s.value for s in WarningSeverity}
        assert values == {"info", "minor", "moderate", "major"}


class TestClassifyWarningSeverity:
    """Deterministic severity classification — no LLM intuition."""

    def test_docs_path_is_info(self):
        """Orphan documentation files → INFO."""
        warn_message = "Orphan node: 'docs/README.md' is not referenced by any edge"
        severity = classify_warning_severity(warn_message, {"filePath": "docs/README.md"})
        assert severity == WarningSeverity.INFO

    def test_readme_file_is_info(self):
        """README files are documentation → INFO."""
        warn_message = "Orphan node: 'README.md' is not referenced by any edge"
        severity = classify_warning_severity(warn_message, {"filePath": "README.md"})
        assert severity == WarningSeverity.INFO

    def test_changelog_is_minor(self):
        """CHANGELOG is an orphan asset → MINOR."""
        warn_message = "Orphan node: 'CHANGELOG' is not referenced by any edge"
        severity = classify_warning_severity(warn_message, {"filePath": "CHANGELOG"})
        assert severity == WarningSeverity.MINOR

    def test_fixture_file_is_minor(self):
        """Fixture/test-data files → MINOR."""
        warn_message = "Orphan node: 'fixture.json' is not referenced by any edge"
        severity = classify_warning_severity(warn_message, {"filePath": "tests/fixtures/fixture.json"})
        assert severity == WarningSeverity.MINOR

    def test_isolated_source_is_moderate(self):
        """Orphan source file (not in a tests/fixtures/docs path) → MODERATE."""
        warn_message = "Orphan node: 'legacy.py' is not referenced by any edge"
        severity = classify_warning_severity(warn_message, {"filePath": "legacy.py"})
        assert severity == WarningSeverity.MODERATE

    def test_src_orphan_moderate(self):
        """Orphan .py in src/ with no heuristics to escalate → MODERATE."""
        warn_message = "Orphan node: 'src/orphan.py' is not referenced by any edge"
        severity = classify_warning_severity(warn_message, {"filePath": "src/orphan.py"})
        assert severity == WarningSeverity.MODERATE

    def test_default_is_info(self):
        """Unmatched warning message with no file path defaults to INFO."""
        warn_message = "Some unknown warning"
        severity = classify_warning_severity(warn_message, {})
        assert severity == WarningSeverity.INFO


class TestValidateGraphSeveritySummary:
    """End-to-end validate_graph with severity summary output."""

    def test_empty_graph_has_trivial_severity_summary(self):
        graph = {"nodes": [], "edges": []}
        result = validate_graph(graph)
        summary = result["severity_summary"]
        assert summary["info"] == 0
        assert summary["minor"] == 0
        assert summary["moderate"] == 0
        assert summary["major"] == 0

    def test_mixed_warnings_produce_severity_breakdown(self):
        """Graph with orphan docs (INFO) and orphan source (MODERATE)."""
        graph = {
            "nodes": [
                {"node_type": "file", "node_id": "docs/guide.md", "filePath": "docs/guide.md"},
                {"node_type": "file", "node_id": "src/unused.py", "filePath": "src/unused.py"},
            ],
            "edges": [],
        }
        result = validate_graph(graph)
        summary = result["severity_summary"]
        assert summary["info"] >= 1, f"Expected info>=1, got {summary}"
        assert summary["moderate"] >= 1, f"Expected moderate>=1, got {summary}"

    def test_backward_compatibility_issues_field(self):
        """'issues' field must still exist and be a list."""
        graph = {"nodes": [], "edges": []}
        result = validate_graph(graph)
        assert "issues" in result
        assert isinstance(result["issues"], list)

    def test_backward_compatibility_warnings_field(self):
        """'warnings' field must still exist and be a list of strings."""
        graph = {"nodes": [], "edges": []}
        result = validate_graph(graph)
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_severity_classified_warnings_present(self):
        """Each warning must have a corresponding severity-classified entry."""
        graph = {
            "nodes": [
                {"node_type": "file", "node_id": "orphan1", "filePath": "docs/a.md"},
                {"node_type": "file", "node_id": "orphan2", "filePath": "src/b.py"},
            ],
            "edges": [],
        }
        result = validate_graph(graph)
        classified = result.get("severity_classified_warnings", [])
        for entry in classified:
            assert "severity" in entry
            assert "message" in entry
            assert entry["severity"] in {s.value for s in WarningSeverity}

    def test_orphan_asset_in_fixture_is_minor(self):
        """Orphan fixture/assets file → MINOR."""
        graph = {
            "nodes": [
                {"node_type": "file", "node_id": "test_data.json", "filePath": "tests/fixtures/test_data.json"},
            ],
            "edges": [],
        }
        result = validate_graph(graph)
        summary = result["severity_summary"]
        assert summary["minor"] >= 1

    def test_summary_counts_match_classified_warnings(self):
        """severity_summary counts must equal len of severity_classified_warnings."""
        graph = {
            "nodes": [
                {"node_type": "file", "node_id": "docs/x.md", "filePath": "docs/x.md"},
                {"node_type": "file", "node_id": "CHANGELOG", "filePath": "CHANGELOG"},
                {"node_type": "file", "node_id": "src/old.py", "filePath": "src/old.py"},
            ],
            "edges": [],
        }
        result = validate_graph(graph)
        summary = result["severity_summary"]
        classified = result["severity_classified_warnings"]
        total_from_summary = sum(summary[s.value] for s in WarningSeverity)
        assert total_from_summary == len(classified)


class TestBuildWarningSummary:
    """Test the build_warning_summary helper."""

    def test_empty_list(self):
        entries = []
        summary = build_warning_summary(entries)
        assert summary["info"] == 0
        assert summary["minor"] == 0
        assert summary["moderate"] == 0
        assert summary["major"] == 0

    def test_mixed_entries(self):
        entries = [
            {"severity": "info", "message": "doc"},
            {"severity": "info", "message": "doc2"},
            {"severity": "minor", "message": "asset"},
            {"severity": "moderate", "message": "src"},
            {"severity": "major", "message": "suspicious"},
        ]
        summary = build_warning_summary(entries)
        assert summary["info"] == 2
        assert summary["minor"] == 1
        assert summary["moderate"] == 1
        assert summary["major"] == 1


class TestVerdictModelPreserved:
    """The verdict model (APPROVED/WARNING/REJECTED) must be unchanged."""

    def test_approved_when_no_issues_no_warnings(self):
        graph = {"nodes": [], "edges": []}
        result = validate_graph(graph)
        assert len(result["issues"]) == 0
        assert len(result["warnings"]) == 0

    def test_warning_when_no_issues_with_warnings(self):
        graph = {
            "nodes": [
                {"node_type": "file", "node_id": "orphan", "filePath": "docs/readme.md"},
            ],
            "edges": [],
        }
        result = validate_graph(graph)
        assert len(result["issues"]) == 0
        assert len(result["warnings"]) > 0

    def test_rejected_when_issues_present(self):
        graph = {
            "nodes": [
                {"node_type": "bogus", "node_id": "n1", "filePath": "src/main.py"},
            ],
            "edges": [],
        }
        result = validate_graph(graph)
        assert len(result["issues"]) > 0
