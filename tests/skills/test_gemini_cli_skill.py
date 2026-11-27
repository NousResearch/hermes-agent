"""Offline contracts for the Gemini CLI skill's discoverability and guidance."""

import re
from pathlib import Path


SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "gemini-cli"
)


def _skill():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    frontmatter, body = text[4:].split("\n---\n", 1)
    # These contracts need only the skill's scalar/inline-list top-level fields.
    fields = dict(re.findall(r"^([a-z_]+): (.+)$", frontmatter, re.MULTILINE))
    return {key: value.strip('\"\'') for key, value in fields.items()}, body


def test_discovery_metadata_is_concise_attributed_and_platform_gated():
    fields, _ = _skill()
    assert fields["name"] == SKILL_DIR.name
    description = fields["description"]
    assert 0 < len(description) <= 60
    assert description.endswith(".") and description.count(".") == 1
    assert re.fullmatch(r"[^,]+ \(@[\w-]+\), Hermes Agent", fields["author"])
    assert fields["version"] and fields["license"]
    # The prescribed shell/tmux procedure requires a POSIX host.
    platforms = {part.strip() for part in fields["platforms"].strip("[]").split(",")}
    assert platforms and platforms <= {"linux", "macos"}


def test_guidance_is_bounded_ordered_and_reference_links_resolve():
    _, body = _skill()
    assert re.search(r"^# .+ Skill$", body, re.MULTILINE)
    assert len(body.splitlines()) <= 220, "Move optional detail to references/"
    assert re.findall(r"^## (.+)$", body, re.MULTILINE) == [
        "When to Use",
        "Prerequisites",
        "How to Run",
        "Quick Reference",
        "Procedure",
        "Pitfalls",
        "Verification",
    ]
    references = re.findall(r"\[[^\]]+\]\((references/[^)#]+)(?:#[^)]*)?\)", body)
    assert references, "Keep setup and version-sensitive detail discoverable"
    for relative in references:
        path = (SKILL_DIR / relative).resolve()
        assert path.is_relative_to(SKILL_DIR.resolve())
        assert path.is_file() and path.read_text(encoding="utf-8").strip()
