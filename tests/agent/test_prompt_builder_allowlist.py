"""Tests for the per-profile skills.allow allowlist in prompt_builder."""
from __future__ import annotations

import logging
import textwrap

import pytest

from agent.prompt_builder import (
    build_skills_system_prompt,
    clear_skills_system_prompt_cache,
)


def _write_skill(base, category, name, description="A short skill desc"):
    d = base / "skills" / category / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n"
    )


def _write_config(base, body=""):
    (base / "config.yaml").write_text(body)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_skills_system_prompt_cache(clear_snapshot=True)
    yield
    clear_skills_system_prompt_cache(clear_snapshot=True)


class TestSkillsAllowlist:
    def test_no_allow_set_includes_everything(self, monkeypatch, tmp_path):
        """Baseline: when skills.allow is unset, all skills load (current behavior)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_skill(tmp_path, "github", "github-issues", "GitHub issues skill")
        _write_skill(tmp_path, "creative", "ascii-art", "ASCII art generator")
        _write_config(tmp_path, "model:\n  default: gpt-5\n")  # no skills key
        out = build_skills_system_prompt()
        assert "github-issues" in out
        assert "ascii-art" in out

    def test_allow_set_filters_to_those_skills_only(self, monkeypatch, tmp_path):
        """skills.allow set with valid entries → only those load."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_skill(tmp_path, "github", "github-issues", "GitHub issues skill")
        _write_skill(tmp_path, "github", "github-pr-workflow", "PR workflow")
        _write_skill(tmp_path, "creative", "ascii-art", "ASCII art")
        cfg = textwrap.dedent("""\
            skills:
              allow:
                - github/github-issues
                - creative/ascii-art
            """)
        _write_config(tmp_path, cfg)
        out = build_skills_system_prompt()
        assert "github-issues" in out
        assert "ascii-art" in out
        assert "github-pr-workflow" not in out

    def test_allow_with_missing_entry_logs_warning_and_skips(
        self, monkeypatch, tmp_path, caplog
    ):
        """Allowlist references a skill not on disk → warning, no crash, valid entries still load."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_skill(tmp_path, "github", "github-issues", "GitHub issues skill")
        cfg = textwrap.dedent("""\
            skills:
              allow:
                - github/github-issues
                - github/does-not-exist
                - bogus/another-missing
            """)
        _write_config(tmp_path, cfg)
        with caplog.at_level(logging.WARNING):
            out = build_skills_system_prompt()
        assert "github-issues" in out
        assert "does-not-exist" not in out
        warn_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("not found on disk" in m for m in warn_msgs), warn_msgs

    def test_allow_within_cap_no_degradation_warning(
        self, monkeypatch, tmp_path, caplog
    ):
        """With a tight allowlist, no 'exceeds ... cap — dropping descriptions' warning fires."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Many skills on disk, but only a small subset allowed
        for i in range(60):
            _write_skill(
                tmp_path,
                f"category{i % 6}",
                f"skill-{i:02d}",
                f"Description {i} " + "x" * 60,
            )
        cfg = textwrap.dedent("""\
            skills:
              allow:
                - category0/skill-00
                - category1/skill-01
                - category2/skill-02
            """)
        _write_config(tmp_path, cfg)
        with caplog.at_level(logging.WARNING):
            out = build_skills_system_prompt()
        msgs = [r.getMessage() for r in caplog.records]
        assert not any("exceeds" in m and "cap" in m for m in msgs), msgs
        assert "skill-00" in out
        assert "skill-59" not in out
