"""Tests for agent/trajectory.py — the trajectory saving utilities."""

from __future__ import annotations

import json

from agent.trajectory import save_trajectory


def test_save_trajectory_positional_filename_still_binds_filename(tmp_path, monkeypatch):
    """A caller passing ``filename`` positionally (the pre-PR 4th argument)
    must get the custom file, not have the value swallowed as ``outcome``."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "custom.jsonl"

    save_trajectory(
        [{"role": "user", "content": "hi"}],
        "test/model",
        True,
        str(target),
    )

    assert target.exists()
    entry = json.loads(target.read_text(encoding="utf-8"))
    assert entry["completed"] is True
    assert "outcome" not in entry


def test_save_trajectory_outcome_is_keyword_only(tmp_path, monkeypatch):
    """``outcome`` is keyword-only — it can never be rebound by a positional
    ``filename`` argument."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "outcome.jsonl"

    save_trajectory(
        [{"role": "user", "content": "hi"}],
        "test/model",
        False,
        str(target),
        outcome=False,
    )

    entry = json.loads(target.read_text(encoding="utf-8"))
    assert entry["completed"] is False
    assert entry["outcome"] is False
