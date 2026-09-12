"""Frontmatter and content contract for the watch-me skill.

Asserts the repo's hardline authoring standards hold for this skill (the
generic skill-manager tests prove nothing about any particular SKILL.md), plus
the few content invariants that keep the skill honest about what the plugin can
actually do.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = REPO_ROOT / "optional-skills" / "creative" / "watch-me" / "SKILL.md"

MAX_DESCRIPTION_CHARS = 60
MAX_SKILL_CONTENT_CHARS = 100_000

# Words review rejects in a description.
MARKETING_WORDS = ("powerful", "comprehensive", "seamless", "advanced", "revolutionary")


@pytest.fixture(scope="module")
def content() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(content: str) -> dict:
    assert content.startswith("---"), "frontmatter must start at byte 0"
    closing = re.search(r"\n---\s*\n", content[3:])
    assert closing is not None, "frontmatter must close with a --- line"
    parsed = yaml.safe_load(content[3 : closing.start() + 3])
    assert isinstance(parsed, dict), "frontmatter must parse as a mapping"
    return parsed


def test_required_fields_present(frontmatter):
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in frontmatter, f"missing frontmatter field: {field}"
    hermes = frontmatter["metadata"]["hermes"]
    assert hermes["tags"]
    assert "related_skills" in hermes


def test_description_meets_the_hardline(frontmatter):
    """60 chars, one sentence, no marketing — the index truncates at 57."""
    description = frontmatter["description"]
    assert len(description) <= MAX_DESCRIPTION_CHARS, f"{len(description)} chars"
    assert description.endswith(".")
    assert description.count(".") == 1, "one sentence"
    lowered = description.lower()
    for word in MARKETING_WORDS:
        assert word not in lowered
    assert frontmatter["name"] not in lowered, "don't repeat the skill name"


def test_body_is_present_and_within_the_size_ceiling(content):
    body = content.split("\n---\n", 1)[1]
    assert body.strip()
    assert len(content) <= MAX_SKILL_CONTENT_CHARS


def test_related_skills_resolve_in_repo(frontmatter):
    """A dangling related_skills entry is a review blocker."""
    for name in frontmatter["metadata"]["hermes"]["related_skills"]:
        found = list((REPO_ROOT / "skills").glob(f"*/{name}")) + list(
            (REPO_ROOT / "optional-skills").glob(f"*/{name}")
        )
        assert found, f"related skill {name!r} does not exist in-repo"


def test_modern_section_order_is_followed(content):
    required = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = []
    for heading in required:
        assert heading in content, f"missing section: {heading}"
        positions.append(content.index(heading))
    assert positions == sorted(positions), "sections are out of the standard order"


def test_no_machine_local_paths(content):
    """An absolute home path baked into a committed skill breaks every other user."""
    assert not re.search(r"/(home|Users)/[a-z0-9_.-]+/", content, re.IGNORECASE)


def test_commands_are_framed_through_hermes_tools(content):
    """CLI-wrapper skills must invoke via the `terminal` tool, not bare shell prose."""
    assert "terminal(command=" in content


def test_platform_gating_matches_the_plugin_support_matrix(frontmatter):
    """The capture layer implements all three hosts, so all three are declared."""
    assert set(frontmatter["platforms"]) == {"linux", "macos", "windows"}


def test_states_the_explicit_capture_boundary(content):
    """The skill must not read as ambient screen watching.

    Live mode exists now, so the boundary is no longer "there is no live mode" —
    it is that capture STARTS AND STOPS ON REQUEST. A skill that implies
    otherwise would have the agent offering surveillance it cannot do and the
    user did not ask for.
    """
    lowered = " ".join(content.lower().split())
    assert "explicit" in lowered
    assert "starts and stops on request" in lowered
    assert "nothing is observed in between" in lowered


def test_warns_that_live_quiet_is_not_live_broken(content):
    """The failure mode a user will hit first, and misread.

    Most seconds produce no model call by design, so a session with two comments
    in ten minutes is healthy. A total capture failure produces the same silence
    — the skill has to say which is which and where to look.
    """
    assert "quiet on purpose" in content.lower()
    assert "Zero frames" in content


def test_warns_that_video_input_is_not_universal(content):
    """Claude rejects video outright — the top failure mode worth pre-empting."""
    assert "Claude does not accept video input" in content


def test_documents_the_counterintuitive_cost_levers(content):
    """Resolution is a bytes lever; retiming is the cost lever.

    Both facts are surprising enough that an agent reasoning from first
    principles gets them wrong, so the skill has to state them.
    """
    assert "Resolution does not change the token bill" in content
    assert "sample video at ~1 fps" in content.replace("**", "")


def test_flags_the_window_title_privacy_control(content):
    """Titles carry documents and URLs; --no-titles must be discoverable."""
    assert "--no-titles" in content
    assert "private" in content.lower()
