from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


SKILLS = {
    "hermes-x-posting-workflows": ROOT
    / "skills"
    / "social-media"
    / "hermes-x-posting-workflows"
    / "SKILL.md",
    "hermes-memory-plugin-integration": ROOT
    / "skills"
    / "software-development"
    / "hermes-memory-plugin-integration"
    / "SKILL.md",
    "open-source-maintainer-applications": ROOT
    / "skills"
    / "github"
    / "open-source-maintainer-applications"
    / "SKILL.md",
    "oss-program-application-strategy": ROOT
    / "skills"
    / "github"
    / "oss-program-application-strategy"
    / "SKILL.md",
}


def _frontmatter_and_body(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    end = text.index("\n---\n", 4)
    raw = text[4:end]
    body = text[end + len("\n---\n") :]
    data: dict[str, str] = {}
    for line in raw.splitlines():
        if not line or line.startswith("  ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data, body


def test_local_workflow_skills_have_short_descriptions_and_core_sections():
    for expected_name, path in SKILLS.items():
        metadata, body = _frontmatter_and_body(path)

        assert metadata["name"] == expected_name
        description = metadata["description"]
        assert description.endswith(".")
        assert len(description) <= 60

        for section in (
            "## When to Use",
            "## Prerequisites",
            "## How to Run",
            "## Quick Reference",
            "## Procedure",
            "## Pitfalls",
            "## Verification",
        ):
            assert section in body


def test_x_posting_workflow_keeps_public_write_safety_rules():
    body = SKILLS["hermes-x-posting-workflows"].read_text(encoding="utf-8")

    assert "explicit user approval" in body
    assert "URL and the final posted text" in body
    assert "Never read or print local credential stores" in body
    assert "stores only public post metadata" in body
    assert "twitter_request_catalog" in body
    assert "reviewed exact `text`" in body
    assert "degraded=false" in body
    assert "`GET` method only" in body
    assert "direct messages, bookmarks, drafts, account settings" in body
    assert "success=false" in body
    assert "cursor-free" in body
    assert "X Premium" in body
    assert "`x_search` is search-only" in body
    assert "Skip `x_search` for posting-only requests" in body
    assert "block a `lm-twitterer` post" in body
    assert "x_search_required_for_posting=false" in body


def test_memory_plugin_skill_links_reference_and_blocks_real_home_writes():
    body = SKILLS["hermes-memory-plugin-integration"].read_text(encoding="utf-8")
    reference = (
        ROOT
        / "skills"
        / "software-development"
        / "hermes-memory-plugin-integration"
        / "references"
        / "lm-twitterer-ebbinghaus-bridge.md"
    )

    assert reference.exists()
    assert "get_hermes_home()" in body
    assert "Do not write tests against the user's real `HERMES_HOME`." in body


def test_oss_application_skills_require_live_evidence():
    maintainer_body = SKILLS["open-source-maintainer-applications"].read_text(
        encoding="utf-8"
    )
    strategy_body = SKILLS["oss-program-application-strategy"].read_text(
        encoding="utf-8"
    )

    assert "re-check current public sources" in maintainer_body
    assert "Every material claim has a URL" in maintainer_body
    assert "Official pages and current repository evidence win" in strategy_body
    assert "Do not quote stale GitHub counts" in strategy_body
