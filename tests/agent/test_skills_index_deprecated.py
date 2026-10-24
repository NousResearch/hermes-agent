"""Guard: the FLAT SKILLS INDEX must stay RETIRED (card t_cc3c6951, operator ruling 2026-09-30).

The index was the `## Skills` block that rendered every visible skill as one
`- name: one-line description` line and told the agent to scan it. It is replaced by the curated
hierarchical surface each lane's SOUL carries (`## Skills — your curated surface`), loaded by name.

This file is the REINTRODUCTION GUARD. It fails when:

  * `_render_skills_index` renders index lines again (someone restored the old code path), or
  * the assembled skills block carries index lines for the real skills tree, or
  * the deprecation banner stops naming the curated surface as the replacement.

Run:  <hermes-venv>/bin/python -m pytest tests/agent/test_skills_index_deprecated.py -q
Red proof (the hatch re-arms the index, and the guard catches it): pin `skills.index: flat` in a
profile's config.yaml — the hatch is a config setting, never a `HERMES_*` env var.
"""

import re

import pytest

from agent import prompt_builder as pb

# An index line looks exactly like `    - yaan-something: Use when ...`
INDEX_LINE_RE = re.compile(r"^\s+- [A-Za-z0-9_.:-]+:", re.MULTILINE)

SAMPLE = {
    "yoyodine": [("yaan-agent-principles", "Use FIRST every task: yoyodine operating principles.")],
    "domain:platform": [("yaan-cron-execution", "Use when writing cron job code or shell steps.")],
}


@pytest.fixture(autouse=True)
def _config_is_silent(monkeypatch):
    """Never let the host's own config.yaml decide this test's verdict."""
    monkeypatch.setattr(pb, "_config_readonly", lambda what: {}, raising=False)


def _config(monkeypatch, skills: dict) -> None:
    """Point the prompt builder at a config.yaml shape for one test."""
    monkeypatch.setattr(pb, "_config_readonly", lambda what: {"skills": skills}, raising=False)


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
    _config(monkeypatch, {"index": "dark"})
    assert pb.skills_index_dark() is True
    _config(monkeypatch, {"index": "flat"})
    assert pb.skills_index_dark() is False
    block = pb._render_skills_index(SAMPLE, {}, None, {"terminal"})
    assert INDEX_LINE_RE.search(block), "the hatch must restore the legacy rendering (red proof)"


def test_config_hatch_is_recognised(monkeypatch):
    _config(monkeypatch, {"index": "flat"})
    assert pb.skills_index_dark() is False


def test_config_default_declares_the_hatch_and_the_loader_honours_it(monkeypatch, tmp_path):
    """The hatch is a REAL setting: declared in DEFAULT_CONFIG, reachable through config.yaml.

    ``skills.index`` was initially read without a ``DEFAULT_CONFIG`` entry — a reader with no
    registry entry, which keeps the key invisible to `hermes config` and unguarded by the
    defaults/consumers invariant. This pins both halves: the declared default is dark, and the same
    value written to a profile's config.yaml reaches ``skills_index_dark`` through the real loader
    (not the stubbed reader this module otherwise uses).
    """
    from hermes_cli.config import DEFAULT_CONFIG, load_config_readonly

    # Declared default: dark, on the same spelling the reader treats as "not flat".
    assert DEFAULT_CONFIG["skills"]["index"] == "dark"
    assert not (DEFAULT_CONFIG["skills"]["index"] is True)

    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text("skills:\n  index: flat\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pb, "_config_readonly", lambda what="": load_config_readonly(), raising=False)
    assert pb.skills_index_dark() is False, "skills.index: flat in config.yaml must reach the reader"


def test_assembled_block_for_a_real_tree_shape_is_index_free(tmp_path):
    """Hermetic end-to-end leg: a real skills-tree shape assembles to the banner, not a list.

    The LIVE tree is deliberately NOT read here — the test harness refuses real-home file I/O
    (tests/home_io_guard.py). The live leg is scripts/check_skills_index_dark.py in yaan-platform,
    which renders every lane home's real block and fails the same way.
    """
    from pathlib import Path

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
