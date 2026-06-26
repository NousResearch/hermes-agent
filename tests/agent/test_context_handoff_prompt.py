"""Tests for injecting latest handoff into a fresh system prompt."""

from __future__ import annotations

from pathlib import Path

from agent.context_handoff import load_latest_context_handoff_prompt


def test_load_latest_context_handoff_prompt_returns_empty_without_file(
    tmp_path: Path, monkeypatch
):
    """No latest handoff means no prompt block and no cache churn from errors."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert load_latest_context_handoff_prompt() == ""


def test_load_latest_context_handoff_prompt_includes_verification_discipline(
    tmp_path: Path, monkeypatch
):
    """Existing handoff is loaded with instructions to verify live state first."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    handoff_dir = tmp_path / "handoffs"
    handoff_dir.mkdir()
    (handoff_dir / "latest.md").write_text(
        "# Hermes Context Handoff\n\n- Reason: `compression_aborted`\n",
        encoding="utf-8",
    )

    prompt = load_latest_context_handoff_prompt()

    assert "Previous-session handoff" in prompt
    assert "compression_aborted" in prompt
    assert "verify current git / PR / CI / deployment state" in prompt
    assert "Do not trust lossy compression summaries alone" in prompt


def test_load_latest_context_handoff_prompt_truncates_large_file(
    tmp_path: Path, monkeypatch
):
    """Huge handoff markdown is capped before prompt injection."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    handoff_dir = tmp_path / "handoffs"
    handoff_dir.mkdir()
    (handoff_dir / "latest.md").write_text("x" * 100, encoding="utf-8")

    prompt = load_latest_context_handoff_prompt(max_chars=10)

    assert "x" * 10 in prompt
    assert "[handoff truncated]" in prompt
    assert "x" * 50 not in prompt
