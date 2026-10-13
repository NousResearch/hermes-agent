"""Guard: the FLAT SKILLS INDEX must stay RETIRED (card t_cc3c6951, operator ruling 2026-09-30).

The index was the `## Skills` block that rendered every visible skill as one
`- name: one-line description` line and told the agent to scan it. It is replaced by the curated
hierarchical surface each lane's SOUL carries (`## Skills — your curated surface`), loaded by name.

This file is the REINTRODUCTION GUARD. It fails when:

  * `_render_skills_index` renders index lines again (someone restored the old code path), or
  * the assembled skills block carries index lines for the real skills tree, or
  * the deprecation banner stops naming the curated surface as the replacement.

Run:  <hermes-venv>/bin/python -m pytest tests/agent/test_skills_index_deprecated.py -q
Red proof (the hatch re-arms the index, and the guard catches it):
      HERMES_SKILLS_INDEX=flat ... -m pytest tests/agent/test_skills_index_deprecated.py -q
"""

import re

import pytest

from agent import prompt_builder as pb

# An index line looks exactly like `    - yaan-something: Use when ...`
INDEX_LINE_RE = re.compile(r"^\s+- [A-Za-z0-9_.:-]+:", re.M)

SAMPLE = {
    "yoyodine": [("yaan-agent-principles", "Use FIRST every task: yoyodine operating principles.")],
    "domain:platform": [("yaan-cron-execution", "Use when writing cron job code or shell steps.")],
}


@pytest.fixture(autouse=True)
def _config_is_silent(monkeypatch):
    """Never let the host's own config.yaml decide this test's verdict."""
    monkeypatch.delenv("HERMES_SKILLS_INDEX", raising=False)
    monkeypatch.setattr(pb, "_config_readonly", lambda what: {}, raising=False)


def test_default_is_dark():
    assert pb.skills_index_dark() is True


def test_renderer_emits_no_index_lines_and_the_banner():
    block = pb._render_skills_index(SAMPLE, {}, None, {"terminal"})
    assert block, "the deprecated block must still say something"
    assert not INDEX_LINE_RE.search(block), f"index lines re-rendered:\n{block}"
    assert "RETIRED" in block
    assert pb.SKILLS_INDEX_REPLACEMENT in block, "the banner must name the replacement surface"
    assert pb.SKILLS_INDEX_DEPRECATED_ON in block, "the banner must carry the deprecation date"


def test_empty_skills_tree_stays_silent():
    assert pb._render_skills_index({}, {}, None, None) == ""


def test_escape_hatch_is_explicit_and_per_profile(monkeypatch):
    """Flat is reachable ONLY through the hatch — and the hatch is what the guard catches."""
    monkeypatch.setenv("HERMES_SKILLS_INDEX", "dark")
    assert pb.skills_index_dark() is True
    monkeypatch.setenv("HERMES_SKILLS_INDEX", "flat")
    assert pb.skills_index_dark() is False
    block = pb._render_skills_index(SAMPLE, {}, None, {"terminal"})
    assert INDEX_LINE_RE.search(block), "the hatch must restore the legacy rendering (red proof)"


def test_config_hatch_is_recognised(monkeypatch):
    monkeypatch.setattr(pb, "_config_readonly", lambda what: {"skills": {"index": "flat"}}, raising=False)
    assert pb.skills_index_dark() is False


def test_assembled_block_for_a_real_tree_shape_is_index_free(tmp_path):
    """Hermetic end-to-end leg: a real skills-tree shape assembles to the banner, not a list.

    The LIVE tree is deliberately NOT read here — the test harness refuses real-home file I/O
    (tests/home_io_guard.py). The live leg is scripts/check_skills_index_dark.py in yaan-platform,
    which renders every lane home's real block and fails the same way.
    """
    import os
    from pathlib import Path

    os.environ.pop("HERMES_SKILLS_INDEX", None)  # noqa: S105 - not a secret, an env key
    skill = tmp_path / "skills" / "yoyodine" / "yaan-example" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: yaan-example\ndescription: \"Use when proving the index stays dark.\"\n---\n\n# x\n",
        encoding="utf-8")
    block = pb.build_skills_system_prompt(
        available_tools={"terminal", "skill_view", "skills_list"},
        skills_dir_override=Path(tmp_path / "skills"))
    assert block, "a populated tree must still render a block"
    assert not INDEX_LINE_RE.search(block), f"index lines rendered for a real tree shape:\n{block}"
    assert pb.SKILLS_INDEX_REPLACEMENT in block
