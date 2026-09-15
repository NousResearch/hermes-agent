"""Specific shipped-document obligations; generic policy lives in test_authoring_standards.

These are document contracts, not proof that an agent obeys the instructions.
Section ordering, step counts and other prose quality remain review-owned.
"""
from pathlib import Path
import re

import pytest

from agent.skill_utils import parse_frontmatter, skill_matches_platform

REPO = Path(__file__).resolve().parents[2]
GITHUB = "skills/software-development/github"
QUESTIONNAIRE = "optional-skills/productivity/decision-questionnaire"
ACTUAL = "optional-skills/devops/actual-setup"
DARWINIAN = "optional-skills/research/darwinian-evolver"
PINECONE = "optional-skills/research/pinecone-research"
EMAIL = "skills/email/email-inbox-triage"
MEETING = "skills/productivity/meeting-action-items"
PRICE = "skills/productivity/product-price-monitor"
WEEKLY = "skills/productivity/weekly-review-planning"
SOCIAL = "optional-skills/creative/social-media-content-calendar"
MCP = "optional-skills/mcp/mcp-oauth-remote-gateway"


def _document(skill):
    return parse_frontmatter((REPO / skill / "SKILL.md").read_text(encoding="utf-8"))


@pytest.mark.parametrize("skill,contributor", [
    (GITHUB, "benbarclay"), (MEETING, "benbarclay"), (PRICE, "benbarclay"),
    (WEEKLY, "benbarclay"), (EMAIL, "benbarclay"), (SOCIAL, "benbarclay"),
    (ACTUAL, "shl0ms"), (DARWINIAN, "Bihruze"),
    (QUESTIONNAIRE, "mattpocock"), (PINECONE, None), (MCP, None),
])
def test_required_document_and_credit(skill, contributor):
    metadata, _ = _document(skill)  # A corpus glob alone cannot catch a deleted skill.
    if contributor:
        assert contributor in metadata["author"]
    if skill != PINECONE:
        assert not metadata["author"].startswith("Hermes Agent")


@pytest.mark.parametrize("skill,references", [
    (GITHUB, ("auth.md", "issues.md", "pr-workflow.md", "issue-to-pr.md",
              "code-review.md", "repo-management.md")),
    (ACTUAL, ("opencode.md",)),
])
def test_required_references_are_present_and_routed(skill, references):
    _, body = _document(skill)
    for reference in references:
        relative = f"references/{reference}"
        assert (REPO / skill / relative).is_file(), relative
        assert relative in body, relative


@pytest.mark.parametrize("skill,required,forbidden", [
    (QUESTIONNAIRE, ("## Context", "## How to answer", "## Anything else?",
                     "decision-questionnaire-<slug>.md", "Interview the Send"),
     ("claude", "slash command", "disable-model-invocation")),
    (ACTUAL, ("--provider actual",), ("hermes config set providers.actual.api ", "key_env")),
    (MEETING, ("never invent", "before creating anything", "`unresolved`"), (r"\bLinear\b",)),
    (PRICE, ("Setup (foreground, once)", "Tick (each scheduled run)", "cronjob(action=",
             "Do not schedule until one foreground fetch works",
             "never overwrite the last good observation", "fingerprint"), ("flight-research",)),
    (EMAIL, ("read + draft", "does not imply permission"), ()),
    (SOCIAL, (), ("image-generation-workflow",)),
])
def test_document_obligations(skill, required, forbidden):
    _, body = _document(skill)
    for text in required:
        assert text in body, text
    for pattern in forbidden:
        assert not re.search(pattern, body, re.I), pattern
    if skill == SOCIAL:
        assert "handed-off, not published" in body or "handed-off slots" in body


@pytest.mark.parametrize("relative", [
    f"{GITHUB}/references/issue-to-pr.md",
    *(f"{skill}/SKILL.md" for skill in (MEETING, PRICE, WEEKLY, EMAIL, SOCIAL)),
])
def test_procedure_steps_have_completion_criteria(relative):
    body = (REPO / relative).read_text(encoding="utf-8")
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## |\Z)", body, re.M | re.S)
    assert steps, relative
    for step in steps:
        assert "Done when" in step, step[:80]
        if relative.endswith("issue-to-pr.md"):
            assert not re.match(r"^### \d+\.[^\n]*\n+Load `", step)


def test_issue_to_pr_disciplines():
    body = (REPO / GITHUB / "references/issue-to-pr.md").read_text(encoding="utf-8")
    for text in ("--comments", "pr list --search", "git log -p -S", "sibling", "dispatches CI"):
        assert text in body, text
    assert "sabotage" in body.lower() or "FAILS" in body
    assert "/home/" not in body and not re.search(r"[A-Z]:\\+Users", body)


def test_optional_platform_declarations():
    darwinian, _ = _document(DARWINIAN)
    assert "windows" not in darwinian["platforms"]  # func_timeout relies on POSIX signals.
    assert {"linux", "macos"} <= set(darwinian["platforms"])
    assert _document(MCP)[0]["platforms"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("skill", [QUESTIONNAIRE, DARWINIAN])
def test_optional_skill_available_on_posix(skill):
    assert skill_matches_platform(_document(skill)[0])


@pytest.mark.platforms("windows")
def test_optional_skill_availability_on_windows():
    assert skill_matches_platform(_document(QUESTIONNAIRE)[0])
    assert not skill_matches_platform(_document(DARWINIAN)[0])
