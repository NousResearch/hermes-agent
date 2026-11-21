"""Shape of the <available_skills> index rows (single-skill categories, headers, clipped descriptions)."""
import re

import pytest

from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache
from agent.skill_utils import SKILL_PROMPT_DESC_LIMIT, clip_skill_description
from agent.skills_index_delta import index_entries

LONG = "Use when cutting, publishing, or abandoning a hermes-agent release and its notes."


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_skills_system_prompt_cache(clear_snapshot=True)
    yield
    clear_skills_system_prompt_cache(clear_snapshot=True)


def _skill(root, rel, desc):
    d = root / rel
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {d.name}\ndescription: {desc}\n---\n\nBody.\n", encoding="utf-8")


def test_index_rows_carry_each_name_once_and_never_a_half_word(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skills = tmp_path / "skills"
    _skill(skills, "creative/ascii-art", "ASCII art banners.")
    _skill(skills, "creative/pixel-art", "Pixel art sprites.")
    (skills / "creative" / "DESCRIPTION.md").write_text(
        "---\ndescription: Creative content generation and visual design tools.\n---\n", encoding="utf-8")
    _skill(skills, "computer-use", "Drive the desktop.")  # a top-level skill is its own category
    _skill(skills, "github/hermes-release", LONG)

    prompt = build_skills_system_prompt()
    block = re.search(r"<available_skills>\n(.*?)\n</available_skills>", prompt, re.DOTALL).group(1)
    lines = block.splitlines()

    assert "  creative:" in lines and "Creative content generation" not in block
    assert "  - computer-use: Drive the desktop." in lines and "  computer-use:" not in lines
    release = next(line for line in lines if "hermes-release" in line)
    assert release.startswith("  - hermes-release: ")
    shown = release.split(": ", 1)[1]
    assert len(shown) <= SKILL_PROMPT_DESC_LIMIT and shown.endswith("...")
    kept = shown[:-3]
    assert LONG.startswith(kept) and LONG[len(kept)] in " ,"  # cut lands between words
    # The turn-delta parser reads flattened rows like nested ones.
    assert set(index_entries(prompt)) == {"ascii-art", "pixel-art", "computer-use", "hermes-release"}


@pytest.mark.parametrize("sep", ["\t", "\u00a0"], ids=["tab", "nbsp"])
def test_clip_cuts_at_any_whitespace_never_inside_a_word(sep):
    desc = f"alpha{sep}" + f"bravo{sep}" * 12
    shown = clip_skill_description(desc)
    assert len(shown) <= SKILL_PROMPT_DESC_LIMIT and shown.endswith("...")
    assert set(shown[:-3].split(sep)) == {"alpha", "bravo"}  # whole words only, no "bra..."



def test_index_desc_limit_widens_the_index_and_rebuilds_a_stale_snapshot(monkeypatch, tmp_path):
    """skills.index_desc_limit shows the full description; flipping it back clips again even though
    no SKILL.md changed (the on-disk snapshot and LRU must not serve rows clipped at the old width)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _skill(tmp_path / "skills", "github/hermes-release", LONG)
    config = tmp_path / "config.yaml"

    def release_row(fresh_process: bool):
        if fresh_process:  # drop only the in-process LRU; the disk snapshot stays
            clear_skills_system_prompt_cache()
        return index_entries(build_skills_system_prompt())["hermes-release"]

    assert release_row(fresh_process=False).endswith("...")
    config.write_text("skills:\n  index_desc_limit: 200\n", encoding="utf-8")
    assert release_row(fresh_process=False) == f"  - hermes-release: {LONG}"  # LRU must not serve the old width
    config.write_text("skills:\n  index_desc_limit: 5\n", encoding="utf-8")  # below the floor: 60
    assert release_row(fresh_process=True).endswith("...")  # nor the disk snapshot
    config.write_text("skills:\n  index_desc_limit: \"\u00b2\"\n", encoding="utf-8")  # isdigit() but not int()
    assert release_row(fresh_process=True).endswith("...")
