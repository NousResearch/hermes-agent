"""Tests for linear_api.py update-issue safety guardrails.

The guardrails added to cmd_update_issue catch the classic shell antipattern
where a stage-1 script writes an error sentinel to stdout (e.g. "FATAL: ...")
on failure, the outer shell does not gate on exit code, and the calling
update-issue --description "$(cat tmp)" silently replaces the issue body with
the error string. Two guardrails:

  1. Pre-write: refuse FATAL-shaped or trivially-short payloads unless
     --allow-truncate is set.
  2. Post-write: re-fetch and emit stderr WARNING if persisted description
     shrank significantly. Skippable via --skip-verify.
"""

import importlib.util
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

API_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/linear/scripts/linear_api.py"
)


@pytest.fixture
def api_module(monkeypatch):
    """Load linear_api.py as a module without invoking its CLI entry point."""
    monkeypatch.setenv("LINEAR_API_KEY", "test-key-not-used")
    spec = importlib.util.spec_from_file_location("linear_api_test", API_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_args(api_module, **overrides):
    """Build a Namespace for cmd_update_issue with sensible defaults."""
    defaults = dict(
        identifier="TEST-1",
        title=None,
        description=None,
        priority=None,
        allow_truncate=False,
        skip_verify=False,
    )
    defaults.update(overrides)
    return api_module.argparse.Namespace(**defaults)


# ── Pre-write guardrail ────────────────────────────────────────────────────

def test_refuses_fatal_prefix(api_module, capsys):
    """A FATAL-prefixed payload trips the error-shape check."""
    args = _make_args(api_module, description="FATAL: ordering section not found verbatim")
    with patch.object(api_module, "gql") as mock_gql:
        with pytest.raises(SystemExit) as exc:
            api_module.cmd_update_issue(args)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    assert "error_shape=True" in captured.err
    # Mutation must not have been attempted
    mock_gql.assert_not_called()


def test_refuses_error_traceback(api_module, capsys):
    """A Traceback-prefixed payload also trips the error-shape check."""
    args = _make_args(
        api_module,
        description="Traceback (most recent call last):\n  File ...",
    )
    with patch.object(api_module, "gql") as mock_gql:
        with pytest.raises(SystemExit) as exc:
            api_module.cmd_update_issue(args)
    assert exc.value.code == 2
    assert "REFUSED" in capsys.readouterr().err
    mock_gql.assert_not_called()


def test_refuses_short_unstructured(api_module, capsys):
    """A 33-char single-line payload with no markdown structure trips the length+structure check."""
    args = _make_args(api_module, description="just a single line of normal text")
    with patch.object(api_module, "gql") as mock_gql:
        with pytest.raises(SystemExit) as exc:
            api_module.cmd_update_issue(args)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    assert "structured=False" in captured.err
    mock_gql.assert_not_called()


def test_accepts_markdown_description(api_module):
    """A real markdown body with multiple lines passes the guardrail."""
    desc = (
        "## Context\n\n"
        "This is a real description with **markdown** structure and multiple lines.\n"
        "It should not be refused by the safety check."
    )
    args = _make_args(api_module, description=desc)
    with patch.object(api_module, "gql") as mock_gql:
        # First call = mutation; second = read-after-write verify
        mock_gql.side_effect = [
            {"issueUpdate": {"success": True, "issue": {"identifier": "TEST-1"}}},
            {"issue": {"description": desc}},
        ]
        api_module.cmd_update_issue(args)
    assert mock_gql.call_count == 2  # mutation + verify


def test_accepts_long_payload_without_markdown(api_module):
    """A 500-char payload passes even without markdown structure (length floor satisfied)."""
    desc = "a" * 500
    args = _make_args(api_module, description=desc)
    with patch.object(api_module, "gql") as mock_gql:
        mock_gql.side_effect = [
            {"issueUpdate": {"success": True, "issue": {"identifier": "TEST-1"}}},
            {"issue": {"description": desc}},
        ]
        api_module.cmd_update_issue(args)
    assert mock_gql.call_count == 2


def test_allow_truncate_bypasses_guardrail(api_module):
    """--allow-truncate lets a short payload through."""
    args = _make_args(api_module, description="short", allow_truncate=True)
    with patch.object(api_module, "gql") as mock_gql:
        mock_gql.side_effect = [
            {"issueUpdate": {"success": True, "issue": {"identifier": "TEST-1"}}},
            {"issue": {"description": "short"}},
        ]
        api_module.cmd_update_issue(args)
    assert mock_gql.call_count == 2  # mutation invoked; verify still runs


def test_allow_truncate_bypasses_error_shape(api_module):
    """--allow-truncate also lets a FATAL-shaped payload through (operator override)."""
    args = _make_args(
        api_module,
        description="FATAL but intentional",
        allow_truncate=True,
    )
    with patch.object(api_module, "gql") as mock_gql:
        mock_gql.side_effect = [
            {"issueUpdate": {"success": True, "issue": {"identifier": "TEST-1"}}},
            {"issue": {"description": "FATAL but intentional"}},
        ]
        api_module.cmd_update_issue(args)
    assert mock_gql.call_count == 2


# ── Post-write read-after-write verify ─────────────────────────────────────

def test_post_write_verify_warns_on_truncation(api_module, capsys):
    """If persisted description is much shorter than sent, emit a stderr WARNING."""
    desc = "## Section\n\n" + ("real content paragraph " * 50)  # ~1100 chars, markdown
    args = _make_args(api_module, description=desc)
    with patch.object(api_module, "gql") as mock_gql:
        # Mutation returns success; verify returns a degenerate persisted body.
        mock_gql.side_effect = [
            {"issueUpdate": {"success": True, "issue": {"identifier": "TEST-1"}}},
            {"issue": {"description": "got truncated somehow"}},
        ]
        api_module.cmd_update_issue(args)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "truncated" in captured.err
    assert "persisted" in captured.err


def test_post_write_verify_silent_on_match(api_module, capsys):
    """When the persisted description matches what was sent, no warning is emitted."""
    desc = "## Section\n\n" + ("real content paragraph " * 50)
    args = _make_args(api_module, description=desc)
    with patch.object(api_module, "gql") as mock_gql:
        mock_gql.side_effect = [
            {"issueUpdate": {"success": True, "issue": {"identifier": "TEST-1"}}},
            {"issue": {"description": desc}},  # full body persisted
        ]
        api_module.cmd_update_issue(args)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.err


def test_skip_verify_bypasses_read_after_write(api_module):
    """--skip-verify suppresses the post-write re-fetch entirely."""
    desc = "## Section\n\n" + ("real content paragraph " * 50)
    args = _make_args(api_module, description=desc, skip_verify=True)
    with patch.object(api_module, "gql") as mock_gql:
        mock_gql.return_value = {
            "issueUpdate": {"success": True, "issue": {"identifier": "TEST-1"}}
        }
        api_module.cmd_update_issue(args)
    # Only the mutation was called; no follow-up query.
    assert mock_gql.call_count == 1


def test_verify_failure_does_not_propagate(api_module, capsys):
    """If the read-after-write query itself raises, the mutation result still emits."""
    desc = "## Section\n\n" + ("real content paragraph " * 50)
    args = _make_args(api_module, description=desc)

    call_count = {"n": 0}

    def fake_gql(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"issueUpdate": {"success": True, "issue": {"identifier": "TEST-1"}}}
        raise RuntimeError("network blip on verify")

    with patch.object(api_module, "gql", side_effect=fake_gql):
        api_module.cmd_update_issue(args)

    captured = capsys.readouterr()
    assert "post-write verify failed" in captured.err
    # The mutation itself still succeeded, so the operation as a whole is OK.


# ── Argparse wiring ────────────────────────────────────────────────────────

def test_argparse_accepts_allow_truncate_and_skip_verify(api_module):
    """The new flags are wired through build_parser()."""
    parser = api_module.build_parser()
    parsed = parser.parse_args([
        "update-issue", "TEST-1",
        "--description", "short",
        "--allow-truncate",
        "--skip-verify",
    ])
    assert parsed.allow_truncate is True
    assert parsed.skip_verify is True


def test_argparse_defaults_are_safe(api_module):
    """allow_truncate and skip_verify default to False (guardrails on)."""
    parser = api_module.build_parser()
    parsed = parser.parse_args([
        "update-issue", "TEST-1",
        "--description", "## hi\n\nworld",
    ])
    assert parsed.allow_truncate is False
    assert parsed.skip_verify is False
