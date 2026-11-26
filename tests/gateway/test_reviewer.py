"""Tests for gateway/reviewer.py.

All tests mock the subprocess boundary so hermes is never actually invoked.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from gateway.reviewer import (
    Finding,
    ReviewVerdict,
    _parse_response,
    _render_prompt,
    review,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fence(payload: dict) -> str:
    """Wrap *payload* in a reviewer-verdict fence."""
    return "```reviewer-verdict\n" + json.dumps(payload) + "\n```"


def _approved_payload(**overrides) -> dict:
    base = {
        "verdict": "APPROVED",
        "findings": [],
        "needs_info": None,
        "summary": "All good.",
    }
    base.update(overrides)
    return base


def _blocked_payload(**overrides) -> dict:
    base = {
        "verdict": "BLOCKED",
        "findings": [
            {"severity": "critical", "file": "src/foo.py:42", "issue": "SQL injection"},
        ],
        "needs_info": None,
        "summary": "Critical issue found.",
    }
    base.update(overrides)
    return base


def _needs_info_payload(**overrides) -> dict:
    base = {
        "verdict": "NEEDS_INFO",
        "findings": [],
        "needs_info": "Please clarify the auth flow.",
        "summary": "Awaiting clarification.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _parse_response — happy paths (each verdict variant)
# ---------------------------------------------------------------------------

class TestParseResponseApproved:
    def test_verdict_is_approved(self):
        raw = "Great work!\n\n" + _make_fence(_approved_payload())
        result = _parse_response(raw)
        assert result.verdict == "APPROVED"
        assert result.parsed_ok is True
        assert result.findings == []
        assert result.needs_info is None
        assert result.summary == "All good."
        assert result.raw_response == raw

    def test_approved_with_info_findings(self):
        payload = _approved_payload(findings=[
            {"severity": "info", "file": "README.md", "issue": "Typo on line 3"},
        ])
        raw = _make_fence(payload)
        result = _parse_response(raw)
        assert result.verdict == "APPROVED"
        assert len(result.findings) == 1
        assert result.findings[0].severity == "info"
        assert result.findings[0].file == "README.md"
        assert result.parsed_ok is True


class TestParseResponseBlocked:
    def test_verdict_is_blocked(self):
        raw = _make_fence(_blocked_payload())
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is True
        assert len(result.findings) == 1
        f: Finding = result.findings[0]
        assert f.severity == "critical"
        assert f.file == "src/foo.py:42"
        assert f.issue == "SQL injection"

    def test_blocked_multiple_findings(self):
        payload = _blocked_payload(findings=[
            {"severity": "critical", "file": "a.py:1", "issue": "Issue A"},
            {"severity": "warning", "file": "b.py:99", "issue": "Issue B"},
        ])
        raw = _make_fence(payload)
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert len(result.findings) == 2
        assert result.findings[0].severity == "critical"
        assert result.findings[1].severity == "warning"


class TestParseResponseNeedsInfo:
    def test_verdict_is_needs_info(self):
        raw = _make_fence(_needs_info_payload())
        result = _parse_response(raw)
        assert result.verdict == "NEEDS_INFO"
        assert result.parsed_ok is True
        assert result.needs_info == "Please clarify the auth flow."
        assert result.summary == "Awaiting clarification."


# ---------------------------------------------------------------------------
# _parse_response — failure cases
# ---------------------------------------------------------------------------

class TestParseResponseMissingBlock:
    def test_no_fence_returns_blocked_not_parsed(self):
        raw = "Some review prose with no verdict block."
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False
        assert "No reviewer-verdict block" in result.summary
        assert result.raw_response == raw

    def test_wrong_fence_tag_name(self):
        raw = "```review-result\n{\"verdict\": \"APPROVED\"}\n```"
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False

    def test_empty_string(self):
        result = _parse_response("")
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False


class TestParseResponseMalformedJson:
    def test_truncated_json(self):
        raw = "```reviewer-verdict\n{\"verdict\": \"APPROVED\"\n```"
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False
        assert "JSON parse error" in result.summary

    def test_not_json_object(self):
        raw = "```reviewer-verdict\n[\"APPROVED\"]\n```"
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False

    def test_plain_text_in_fence(self):
        raw = "```reviewer-verdict\nnot json at all\n```"
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False


class TestParseResponseInvalidValues:
    def test_invalid_verdict_enum(self):
        payload = _approved_payload(verdict="MAYBE")
        raw = _make_fence(payload)
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False
        assert "Invalid verdict" in result.summary

    def test_invalid_finding_severity(self):
        payload = _approved_payload(findings=[
            {"severity": "blocker", "file": "x.py", "issue": "Bad"},
        ])
        raw = _make_fence(payload)
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False

    def test_findings_not_a_list(self):
        payload = _approved_payload(findings="bad")
        raw = _make_fence(payload)
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False

    def test_finding_not_a_dict(self):
        payload = _approved_payload(findings=["not-a-dict"])
        raw = _make_fence(payload)
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False


# ---------------------------------------------------------------------------
# _parse_response — extra prose around the fence
# ---------------------------------------------------------------------------

class TestParseResponseWithProse:
    def test_prose_before_fence(self):
        raw = (
            "Here is my detailed review...\n\n"
            "The code has several concerns.\n\n"
            + _make_fence(_approved_payload())
        )
        result = _parse_response(raw)
        assert result.verdict == "APPROVED"
        assert result.parsed_ok is True

    def test_prose_after_fence(self):
        raw = (
            _make_fence(_blocked_payload())
            + "\n\nPlease address these before merging."
        )
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is True

    def test_prose_before_and_after(self):
        raw = (
            "Summary: multiple issues.\n\n"
            + _make_fence(_blocked_payload())
            + "\n\nFeel free to ask questions."
        )
        result = _parse_response(raw)
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is True

    def test_uses_last_fence_when_multiple(self):
        """If the reviewer outputs multiple verdict blocks, we use the last."""
        first = _make_fence(_blocked_payload())
        last = _make_fence(_approved_payload(summary="Final verdict"))
        raw = first + "\n\nActually, on reflection:\n\n" + last
        result = _parse_response(raw)
        assert result.verdict == "APPROVED"
        assert result.summary == "Final verdict"
        assert result.parsed_ok is True


# ---------------------------------------------------------------------------
# _render_prompt
# ---------------------------------------------------------------------------

class TestRenderPrompt:
    def test_includes_artifact_kind(self):
        prompt = _render_prompt("pr", paths=None, diff=None, branch=None, context=None)
        assert "pr" in prompt
        assert "reviewer-verdict" in prompt

    def test_includes_branch(self):
        prompt = _render_prompt("branch", paths=None, diff=None, branch="feat/x", context=None)
        assert "feat/x" in prompt

    def test_includes_paths(self):
        prompt = _render_prompt("files", paths=["a.py", "b.py"], diff=None, branch=None, context=None)
        assert "a.py" in prompt
        assert "b.py" in prompt

    def test_includes_diff(self):
        diff = "- old line\n+ new line"
        prompt = _render_prompt("files", paths=None, diff=diff, branch=None, context=None)
        assert "- old line" in prompt
        assert "+ new line" in prompt

    def test_truncates_large_diff(self):
        big_diff = "x" * 50_000
        prompt = _render_prompt("pr", paths=None, diff=big_diff, branch=None, context=None)
        assert "truncated" in prompt

    def test_includes_context(self):
        prompt = _render_prompt("pr", paths=None, diff=None, branch=None, context="ticket: ABC-123")
        assert "ticket: ABC-123" in prompt

    def test_no_extra_sections_when_none(self):
        prompt = _render_prompt("pr", paths=None, diff=None, branch=None, context=None)
        # Should not crash and should still contain instructions
        assert "reviewer-verdict" in prompt


# ---------------------------------------------------------------------------
# review() — mocked subprocess boundary
# ---------------------------------------------------------------------------

_FAKE_APPROVED_RESPONSE = (
    "Looking good overall.\n\n"
    + _make_fence(_approved_payload(summary="Passes review."))
)

_FAKE_BLOCKED_RESPONSE = (
    "There is a critical issue.\n\n"
    + _make_fence(_blocked_payload(summary="Must fix before merge."))
)


class TestReviewMocked:
    def test_approved_verdict_end_to_end(self):
        with patch("gateway.reviewer._invoke_profile", return_value=_FAKE_APPROVED_RESPONSE):
            result = review("pr", diff="+ foo = 1")
        assert result.verdict == "APPROVED"
        assert result.parsed_ok is True
        assert result.summary == "Passes review."

    def test_blocked_verdict_end_to_end(self):
        with patch("gateway.reviewer._invoke_profile", return_value=_FAKE_BLOCKED_RESPONSE):
            result = review("branch", branch="feat/bad-code")
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is True
        assert len(result.findings) == 1

    def test_needs_info_end_to_end(self):
        fake = "Need more info.\n\n" + _make_fence(_needs_info_payload())
        with patch("gateway.reviewer._invoke_profile", return_value=fake):
            result = review("files", paths=["gateway/foo.py"])
        assert result.verdict == "NEEDS_INFO"
        assert result.needs_info == "Please clarify the auth flow."
        assert result.parsed_ok is True

    def test_subprocess_error_returns_blocked(self):
        import subprocess
        with patch(
            "gateway.reviewer._invoke_profile",
            side_effect=subprocess.CalledProcessError(1, "hermes"),
        ):
            result = review("pr")
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False
        assert "Reviewer process error" in result.summary

    def test_file_not_found_returns_blocked(self):
        with patch(
            "gateway.reviewer._invoke_profile",
            side_effect=FileNotFoundError("hermes not found"),
        ):
            result = review("pr")
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False

    def test_default_profile_is_h2reviewer(self):
        """Verify the default profile arg is passed through to _invoke_profile."""
        calls: list[tuple] = []

        def fake_invoke(profile: str, prompt: str) -> str:
            calls.append((profile, prompt))
            return _FAKE_APPROVED_RESPONSE

        with patch("gateway.reviewer._invoke_profile", side_effect=fake_invoke):
            review("pr")

        assert calls[0][0] == "h2reviewer"

    def test_custom_profile_is_forwarded(self):
        calls: list[tuple] = []

        def fake_invoke(profile: str, prompt: str) -> str:
            calls.append((profile, prompt))
            return _FAKE_APPROVED_RESPONSE

        with patch("gateway.reviewer._invoke_profile", side_effect=fake_invoke):
            review("pr", profile="custom-reviewer")

        assert calls[0][0] == "custom-reviewer"

    def test_raw_response_preserved_on_parse_failure(self):
        junk = "The model hallucinated and returned no verdict block."
        with patch("gateway.reviewer._invoke_profile", return_value=junk):
            result = review("pr")
        assert result.verdict == "BLOCKED"
        assert result.parsed_ok is False
        assert result.raw_response == junk
