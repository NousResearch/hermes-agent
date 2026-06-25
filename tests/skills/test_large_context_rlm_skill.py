from pathlib import Path

from agent.skill_utils import parse_frontmatter


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills/research/large-context-rlm/SKILL.md"


def _read_skill() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_large_context_rlm_skill_exists_with_valid_frontmatter():
    content = _read_skill()
    frontmatter, body = parse_frontmatter(content)

    assert frontmatter["name"] == "large-context-rlm"
    assert "large" in frontmatter["description"].lower()
    assert len(frontmatter["description"]) <= 1024
    assert body.strip()


def test_large_context_rlm_skill_defines_activation_triggers_and_no_paste_rule():
    body = _read_skill().lower()

    for trigger in ["large files", "massive diffs", "long documents", "many files"]:
        assert trigger in body
    assert "do not paste" in body
    assert "recursive_context.create" in body


def test_large_context_rlm_skill_defines_end_to_end_orchestration_loop():
    body = _read_skill().lower()

    expected_steps = [
        "create corpus",
        "search first",
        "read bounded windows",
        "map chunks",
        "delegate_task",
        "synthesize",
        "verify citations",
    ]
    for step in expected_steps:
        assert step in body


def test_large_context_rlm_skill_has_quality_gates_and_failure_modes():
    body = _read_skill().lower()

    for phrase in [
        "citation contract",
        "claim ledger",
        "coverage check",
        "source-line",
        "failure modes",
        "redaction",
    ]:
        assert phrase in body
