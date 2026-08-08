"""Tests for project-state append helper (UA-006).

TDD phases:
  RED   — fixture ledger exists but is not updated (no helper yet).
  GREEN — helper appends a compact UA section, preserving existing content byte-for-byte.
  FULL  — all code_scan tests pass.
"""
import copy
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

# Match existing code_scan test convention: add scripts/code-scan to sys.path
# so we can import the helper directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "code-scan"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# We import the module-under-test.  This will raise ImportError during RED phase.
from project_state_append import (  # noqa: E402
    PROJECT_STATE_LEDGER_NAME,
    append_project_state,
    _build_ua_section,
    _find_ledger_path,
    _normalize_eof,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "project_state"


@pytest.fixture(scope="module")
def fixture_ledger_bytes():
    """Raw bytes of the pre-existing fixture ledger (for preservation check)."""
    path = FIXTURE_DIR / "with_ledger" / ".hermes" / PROJECT_STATE_LEDGER_NAME
    return path.read_bytes()


@pytest.fixture
def tmp_with_ledger(tmp_path, fixture_ledger_bytes):
    """Temp copy of the with_ledger fixture so tests mutate safely."""
    src = FIXTURE_DIR / "with_ledger"
    dest = tmp_path / "with_ledger"
    shutil.copytree(src, dest)
    return dest


@pytest.fixture
def tmp_empty_ledger(tmp_path):
    """Temp copy of the empty_ledger fixture."""
    src = FIXTURE_DIR / "empty_ledger"
    dest = tmp_path / "empty_ledger"
    shutil.copytree(src, dest)
    return dest


@pytest.fixture
def tmp_without_ledger(tmp_path):
    """Temp copy of the without_ledger fixture (no .hermes/PROJECT_STATE.md)."""
    src = FIXTURE_DIR / "without_ledger"
    dest = tmp_path / "without_ledger"
    shutil.copytree(src, dest)
    return dest


# ---------------------------------------------------------------------------
# Sample manifest dict mimicking real run_ua.py output
# ---------------------------------------------------------------------------

SAMPLE_MANIFEST = {
    "run_id": "def456",
    "mode": "structure",
    "target_path": "/some/project",
    "bundle_dir": "/some/bundle",
    "artifact_paths": {
        "scan.json": "/some/bundle/scan.json",
        "graph.json": "/some/bundle/graph.json",
    },
}

SAMPLE_SCAN = {
    "total_files": 42,
    "total_lines": 3500,
    "languages": {
        "python": 30,
        "javascript": 8,
        "markdown": 2,
        "yaml": 1,
        "json": 1,
    },
}

SAMPLE_GRAPH = {
    "summary": {
        "total_nodes": 120,
        "total_edges": 95,
    },
}

SAMPLE_VALIDATION = {
    "issues": ["orphan: src/unused.py"],
    "warnings": ["orphan: README.md"],
}

SAMPLE_CONTEXT = {
    "validation": {
        "verdict": "issues_found",
    },
}


def _make_result_dict():
    """Build a realistic dict of results to pass to append_project_state."""
    return {
        "manifest": copy.deepcopy(SAMPLE_MANIFEST),
        "scan": copy.deepcopy(SAMPLE_SCAN),
        "graph": copy.deepcopy(SAMPLE_GRAPH),
        "validation": copy.deepcopy(SAMPLE_VALIDATION),
        "context": copy.deepcopy(SAMPLE_CONTEXT),
    }


# ===================================================================
# RED PHASE — these should fail before the helper exists
# ===================================================================

class TestModuleImports:
    """Verify the module can be imported and exposes required symbols."""

    def test_project_state_ledger_name(self):
        assert PROJECT_STATE_LEDGER_NAME == "PROJECT_STATE.md"

    def test_append_project_state_callable(self):
        assert callable(append_project_state)

    def test_build_ua_section_callable(self):
        assert callable(_build_ua_section)

    def test_find_ledger_path_callable(self):
        assert callable(_find_ledger_path)


class TestFindLedgerPath:
    """Test ledger detection logic."""

    def test_finds_ledger_when_present(self, tmp_with_ledger):
        found = _find_ledger_path(str(tmp_with_ledger))
        assert found is not None
        assert found.endswith("PROJECT_STATE.md")
        assert os.path.isfile(found)

    def test_returns_none_without_ledger(self, tmp_without_ledger):
        found = _find_ledger_path(str(tmp_without_ledger))
        assert found is None

    def test_returns_none_empty_ledger(self, tmp_empty_ledger):
        # Even empty ledger should be found (the file exists)
        found = _find_ledger_path(str(tmp_empty_ledger))
        assert found is not None


class TestAppendNoLedger:
    """When no PROJECT_STATE.md exists, no state is recorded."""

    def test_returns_project_state_recorded_false(self, tmp_without_ledger):
        results = _make_result_dict()
        result = append_project_state(results, str(tmp_without_ledger))
        assert isinstance(result, dict)
        assert result["project_state_recorded"] is False
        assert result.get("ledger_path") is None

    def test_no_new_files_created(self, tmp_without_ledger):
        results = _make_result_dict()
        append_project_state(results, str(tmp_without_ledger))
        hermes_dir = tmp_without_ledger / ".hermes"
        assert not hermes_dir.exists() or not (
            hermes_dir / PROJECT_STATE_LEDGER_NAME
        ).exists()


# ===================================================================
# GREEN PHASE — helper must preserve and append correctly
# ===================================================================

class TestLedgerPreservation:
    """Existing ledger content must be byte-for-byte preserved before append."""

    def test_preserves_existing_content(self, tmp_with_ledger, fixture_ledger_bytes):
        """Existing content is byte-for-byte identical before the appended section."""
        results = _make_result_dict()
        result = append_project_state(results, str(tmp_with_ledger))

        assert result["project_state_recorded"] is True
        ledger_path = result["ledger_path"]
        assert ledger_path is not None
        assert os.path.isfile(ledger_path)

        new_bytes = Path(ledger_path).read_bytes()
        # The original bytes must appear at the start of the new file
        assert new_bytes.startswith(fixture_ledger_bytes), (
            "Existing ledger content was modified or overwritten"
        )

    def test_appends_ua_section(self, tmp_with_ledger, fixture_ledger_bytes):
        """After append, file is longer and contains new UA section."""
        results = _make_result_dict()
        append_project_state(results, str(tmp_with_ledger))

        ledger_path = _find_ledger_path(str(tmp_with_ledger))
        new_content = Path(ledger_path).read_text()
        new_bytes = Path(ledger_path).read_bytes()

        assert len(new_bytes) > len(fixture_ledger_bytes), (
            "New file should be longer after append"
        )
        # Check that our run_id appears somewhere after the original content
        original_len = len(fixture_ledger_bytes)
        appended_portion = new_bytes[original_len:].decode("utf-8", errors="replace")
        assert "def456" in appended_portion, (
            "Appended section should contain the UA run_id"
        )

    def test_multiple_appends_accumulate(self, tmp_with_ledger, fixture_ledger_bytes):
        """Calling append twice should produce two UA sections."""
        r1 = _make_result_dict()
        r2 = _make_result_dict()
        r2["manifest"]["run_id"] = "ghi789"

        append_project_state(r1, str(tmp_with_ledger))
        append_project_state(r2, str(tmp_with_ledger))

        ledger_path = _find_ledger_path(str(tmp_with_ledger))
        content = Path(ledger_path).read_text()

        # Both run_ids should appear
        assert "def456" in content
        assert "ghi789" in content
        # Original content still at start
        assert content.startswith(fixture_ledger_bytes.decode("utf-8"))


class TestUASectionContent:
    """The appended UA section must contain only compact deterministic info."""

    def test_section_contains_required_fields(self, tmp_with_ledger):
        """All deterministic fields from spec must appear."""
        results = _make_result_dict()
        append_project_state(results, str(tmp_with_ledger))

        ledger_path = _find_ledger_path(str(tmp_with_ledger))
        content = Path(ledger_path).read_text()

        # Required fields per spec:
        expected_markers = [
            "run_id",
            "mode",
            "target_path",
            "artifact_bundle_path",
            "validation_verdict",
            "issue_count",
            "warning_count",
            "file_count",
            "graph_nodes",
            "graph_edges",
            "next_recommended_action",
        ]
        for marker in expected_markers:
            assert marker in content, f"Missing field: {marker}"

    def test_section_contains_top_languages(self, tmp_with_ledger):
        """Top 5 languages must be listed."""
        results = _make_result_dict()
        append_project_state(results, str(tmp_with_ledger))

        ledger_path = _find_ledger_path(str(tmp_with_ledger))
        content = Path(ledger_path).read_text()

        # Should contain language info
        assert "python" in content
        assert "javascript" in content

    def test_section_no_huge_json_blobs(self, tmp_with_ledger):
        """No large JSON blobs in the ledger; should link artifact path instead."""
        results = _make_result_dict()
        append_project_state(results, str(tmp_with_ledger))

        ledger_path = _find_ledger_path(str(tmp_with_ledger))
        content = Path(ledger_path).read_text()

        # No JSON object syntax for raw language data (e.g. '"languages": { ... }').
        assert '"languages":' not in content, (
            "Full languages JSON dict should not be embedded in the ledger"
        )
        # Flat top_5_languages listing of language names is fine (and expected).
        assert "python" in content, (
            "top_5_languages flat listing should include python"
        )
        # Should reference bundle path
        assert "artifact_bundle_path" in content

    def test_severity_counts_recorded(self, tmp_with_ledger):
        """Issue and warning severity counts must appear."""
        results = _make_result_dict()
        append_project_state(results, str(tmp_with_ledger))

        ledger_path = _find_ledger_path(str(tmp_with_ledger))
        content = Path(ledger_path).read_text()

        assert "issue_count" in content
        assert "warning_count" in content


class TestBuildUASection:
    """Unit tests for the section builder (pure function)."""

    def test_returns_nonempty_string(self):
        results = _make_result_dict()
        section = _build_ua_section(results)
        assert isinstance(section, str)
        assert len(section) > 0

    def test_contains_run_id(self):
        results = _make_result_dict()
        section = _build_ua_section(results)
        assert "def456" in section

    def test_contains_mode(self):
        results = _make_result_dict()
        section = _build_ua_section(results)
        assert "structure" in section

    def test_deterministic_output(self):
        """Same input must produce identical output (deterministic)."""
        results = _make_result_dict()
        s1 = _build_ua_section(results)
        s2 = _build_ua_section(copy.deepcopy(results))
        assert s1 == s2

    def test_empty_results_handles_gracefully(self):
        """Minimal results dict should not crash."""
        results = {"manifest": {"run_id": "none", "mode": "unknown",
                                 "target_path": "/none", "bundle_dir": "/none"}}
        section = _build_ua_section(results)
        assert isinstance(section, str)
        assert len(section) > 0


# ===================================================================
# UA-P1-004: RED PHASE — new requirements must fail before implementation
# ===================================================================

class TestEOFNormalization:
    """UA-P1-004: EOF whitespace must be normalized before append."""

    def test_normalize_eof_multiple_trailing_newlines(self):
        """Content ending with multiple newlines should end with exactly one."""
        content = "hello\n\n\n"
        result = _normalize_eof(content)
        assert result == "hello\n"

    def test_normalize_eof_already_single_newline(self):
        """Content that already ends with one newline should be unchanged."""
        content = "hello\n"
        result = _normalize_eof(content)
        assert result == "hello\n"

    def test_normalize_eof_no_trailing_newline(self):
        """Content without trailing newline should get one."""
        content = "hello"
        result = _normalize_eof(content)
        assert result == "hello\n"

    def test_normalize_eof_only_whitespace(self):
        """Content with blank lines at end should normalize to one newline."""
        content = "hello\n\n\n\n"
        result = _normalize_eof(content)
        assert result == "hello\n"

    def test_eof_normalization_applied_during_append(self, tmp_with_ledger, fixture_ledger_bytes):
        """append_project_state should normalize EOF before appending.

        This ensures that existing content ending with multiple blank lines
        gets normalized to exactly one newline before the new section is appended."""
        ledger_path = str(tmp_with_ledger / ".hermes" / PROJECT_STATE_LEDGER_NAME)
        # Add multiple trailing newlines to simulate messy EOF
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write("\n\n\n")

        original_content = Path(ledger_path).read_text()
        # Verify it currently has extra trailing newlines
        assert original_content.endswith("\n\n\n\n"), \
            f"Expected 4 trailing newlines, got content ending with: {repr(original_content[-10:])}"

        results = _make_result_dict()
        append_project_state(results, str(tmp_with_ledger))

        new_content = Path(ledger_path).read_text()
        # After append, the transition between old content and new section should
        # have exactly one blank line (EOF normalized, then blank line prepended to section)
        # The original content's trailing blank lines should be normalized away
        assert "def456" in new_content, "UA run_id should appear after append"


class TestRuntimeReadinessSummary:
    """UA-P1-004: Runtime readiness summary fields should be recorded."""

    def test_verification_status_in_section(self):
        """verification_status from runtime_readiness should appear in section."""
        results = _make_result_dict()
        results["runtime_readiness"] = {
            "verification_status": "verification_blocked",
            "blockers": ["go command not found", "docker unavailable"],
        }
        section = _build_ua_section(results)
        assert "verification_status" in section
        assert "verification_blocked" in section

    def test_top_blockers_capped_to_three(self):
        """Blockers should be listed but capped to 3."""
        results = _make_result_dict()
        results["runtime_readiness"] = {
            "verification_status": "verification_blocked",
            "blockers": ["block1", "block2", "block3", "block4", "block5"],
        }
        section = _build_ua_section(results)
        # Should contain the blocker heading
        assert "blockers" in section
        # block1, block2, block3 should be present
        assert "block1" in section
        assert "block2" in section
        assert "block3" in section
        # block4 and block5 should NOT appear (capped to 3)
        assert "block4" not in section
        assert "block5" not in section

    def test_blockers_absent_when_none(self):
        """No blockers should not produce a blockers field."""
        results = _make_result_dict()
        results["runtime_readiness"] = {
            "verification_status": "verification_ready",
            "blockers": [],
        }
        section = _build_ua_section(results)
        assert "verification_status" in section
        assert "blockers" not in section

    def test_runtime_readiness_in_ledger(self, tmp_with_ledger):
        """When runtime_readiness is provided, it should appear in the ledger."""
        results = _make_result_dict()
        results["runtime_readiness"] = {
            "verification_status": "verification_ready",
            "blockers": [],
        }
        append_project_state(results, str(tmp_with_ledger))
        content = Path(_find_ledger_path(str(tmp_with_ledger))).read_text()
        assert "verification_status" in content
        assert "verification_ready" in content


class TestCleanlinessSummary:
    """UA-P1-004: Target cleanliness summary fields should be recorded."""

    def test_cleanliness_status_in_section(self):
        """target_cleanliness_status from cleanliness should appear in section."""
        results = _make_result_dict()
        results["cleanliness"] = {
            "target_cleanliness_status": "clean",
            "unexpected_changes_count": 0,
        }
        section = _build_ua_section(results)
        assert "target_cleanliness_status" in section
        assert "clean" in section

    def test_unexpected_changes_count_in_section(self):
        """unexpected_changes_count should appear in section."""
        results = _make_result_dict()
        results["cleanliness"] = {
            "target_cleanliness_status": "mutated",
            "unexpected_changes_count": 2,
        }
        section = _build_ua_section(results)
        assert "unexpected_changes_count" in section
        assert "2" in section


class TestNonFatalAppendErrors:
    """UA-P1-004: Append errors must not crash, must record status + error."""

    def test_append_status_success_on_success(self, tmp_with_ledger):
        """Successful append should return project_state_append_status: success."""
        results = _make_result_dict()
        status = append_project_state(results, str(tmp_with_ledger))
        assert status["project_state_append_status"] == "success", \
            f"Expected 'success', got {status.get('project_state_append_status')}"

    def test_append_status_not_attempted_without_ledger(self, tmp_without_ledger):
        """When no ledger exists, status should be not_attempted."""
        results = _make_result_dict()
        status = append_project_state(results, str(tmp_without_ledger))
        assert status["project_state_append_status"] == "not_attempted", \
            f"Expected 'not_attempted', got {status.get('project_state_append_status')}"
        assert status.get("project_state_append_error") is None

    def test_append_status_failed_on_exception(self, tmp_path):
        """When append raises, status should be 'failed' with sanitized error."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        hermes = project_dir / ".hermes"
        hermes.mkdir()
        ledger = hermes / "PROJECT_STATE.md"
        ledger.write_text("# Existing\n")
        ledger.chmod(0o444)  # read-only — append will fail

        try:
            results = _make_result_dict()
            status = append_project_state(results, str(project_dir))
            assert status["project_state_append_status"] == "failed"
            assert "project_state_append_error" in status
            assert status["project_state_append_error"] is not None
            assert len(status["project_state_append_error"]) > 0
        finally:
            ledger.chmod(0o644)  # restore for cleanup
