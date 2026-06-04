#!/usr/bin/env python3
"""
Focused test suite for Drumbeat pivot features.
Tests source-grounding, meta-preamble stripping, and quality gates.
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the drumbeat scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

import draft


def test_strip_meta_preamble():
    """Test that meta-preamble patterns are stripped."""
    # Test with separator
    text_with_separator = """I don't have access to the article.
---
This is the actual content."""
    result = draft.strip_meta_preamble(text_with_separator)
    assert result == "This is the actual content."

    # Test patterns without separator
    text_with_patterns = """I don't have the full article.
Let me draft based on the title.
Here's the actual content that should remain."""
    result = draft.strip_meta_preamble(text_with_patterns)
    assert "I don't have" not in result
    assert "Let me draft" not in result
    assert "Here's the actual content" in result

    print("✓ Meta-preamble stripping works")


def test_quality_gates_hn_link():
    """Test that quality gates detect HN discussion links."""
    # Mock the YAML loading
    mock_gates = {
        "criteria": [
            {
                "id": "citation-hygiene",
                "checks": [
                    {
                        "type": "deterministic",
                        "pattern": r"news\.ycombinator\.com/item\?id=",
                        "fail_if_present": True,
                        "reason": "HN discussion link found"
                    }
                ]
            }
        ]
    }

    candidate = draft.Candidate(
        id=1, source="hn", source_id="12345", url="http://example.com",
        title="Test", summary="Test", raw_json="", fetched_at=0,
        engagement_velocity=1.0, points=100, age_hours=1.0
    )

    with patch.object(draft, 'load_quality_gates', return_value=mock_gates):
        # Test with HN link - should fail
        post_with_hn = "Check out this discussion: https://news.ycombinator.com/item?id=12345"
        result = draft.check_quality_gates(post_with_hn, candidate)
        assert not result.passed
        assert len(result.failures) > 0
        assert "citation-hygiene" in result.failures[0]

        # Test without HN link - should pass
        post_without_hn = "This is a clean post with no HN links"
        result = draft.check_quality_gates(post_without_hn, candidate)
        assert result.passed
        assert len(result.failures) == 0

    print("✓ Quality gates detect HN links correctly")


def test_source_grounding_gate():
    """Test that source-grounding gate fetches content before drafting."""
    # Mock subprocess to simulate successful fetch
    mock_proc = Mock()
    mock_proc.returncode = 0
    mock_proc.stdout = "This is the article content from the web. " * 10  # > 100 chars
    mock_proc.stderr = ""

    with patch('subprocess.run', return_value=mock_proc):
        success, content = draft.fetch_article_content("https://example.com/article")
        assert success
        assert len(content) >= 100
        assert "article content" in content

    # Test failure case - short content
    mock_proc.stdout = "error"
    with patch('subprocess.run', return_value=mock_proc):
        success, reason = draft.fetch_article_content("https://example.com/article")
        assert not success
        assert "too short" in reason

    print("✓ Source-grounding gate works")


def test_quality_gates_blocking():
    """Test that quality gate failures prevent draft from being saved."""
    candidate = draft.Candidate(
        id=1, source="hn", source_id="12345", url="http://example.com",
        title="Test", summary="Test", raw_json="", fetched_at=0,
        engagement_velocity=1.0, points=100, age_hours=1.0
    )

    # Mock quality gate to fail
    with patch.object(draft, 'check_quality_gates') as mock_check:
        mock_check.return_value = draft.QualityGateResult(
            passed=False,
            failures=["[test] Quality gate failure"]
        )

        # Mock connection
        mock_conn = MagicMock()

        try:
            draft.write_draft(mock_conn, candidate, "Test post text", "v1")
            assert False, "Should have raised DrumbeatError"
        except draft.DrumbeatError as exc:
            assert "Quality gate FAILED" in str(exc)
            # Verify that database insert was NOT called
            mock_conn.execute.assert_not_called()

    print("✓ Quality gates block draft save on failure")


def test_skip_candidate():
    """Test that skip_candidate marks candidate as skipped with reason."""
    mock_conn = MagicMock()
    candidate = draft.Candidate(
        id=1, source="hn", source_id="12345", url="http://example.com",
        title="Test", summary="Test", raw_json="", fetched_at=0,
        engagement_velocity=1.0, points=100, age_hours=1.0
    )

    draft.skip_candidate(mock_conn, candidate, "test skip reason")

    # Verify UPDATE was called with correct arguments
    mock_conn.execute.assert_called_once()
    call_args = mock_conn.execute.call_args
    assert "UPDATE candidates" in call_args[0][0]
    assert "skip_reason" in call_args[0][0]
    assert call_args[0][1] == ("test skip reason", 1)

    print("✓ Skip candidate functionality works")


def test_ensure_skip_reason_column():
    """Test idempotent column creation."""
    # Mock connection that doesn't have skip_reason column
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        (0, "id", "INTEGER", 0, None, 1),
        (1, "status", "TEXT", 0, "'new'", 0)
    ]
    mock_conn.execute.return_value = mock_cursor

    draft.ensure_skip_reason_column(mock_conn)

    # Verify PRAGMA and ALTER were called
    calls = [str(call) for call in mock_conn.execute.call_args_list]
    assert any("PRAGMA table_info" in call for call in calls)
    assert any("ALTER TABLE" in call for call in calls)

    print("✓ Skip reason column migration works")


def run_tests():
    """Run all tests."""
    print("Running Drumbeat pivot tests...\n")

    try:
        test_strip_meta_preamble()
        test_quality_gates_hn_link()
        test_source_grounding_gate()
        test_quality_gates_blocking()
        test_skip_candidate()
        test_ensure_skip_reason_column()

        print("\n✓ All tests passed!")
        return 0
    except AssertionError as exc:
        print(f"\n✗ Test failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as exc:
        print(f"\n✗ Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
