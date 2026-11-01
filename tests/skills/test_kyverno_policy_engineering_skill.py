"""Static invariants for the kyverno-policy-engineering optional skill.

The skill ships no scripts, so these tests pin the document contract: modern
section order, native-tool prose, and examples on the CEL policy API rather
than the deprecated ``kyverno.io/v1`` ``ClusterPolicy`` generation.
"""
import re
from pathlib import Path

import hermes_yaml as yaml

SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "devops"
    / "kyverno-policy-engineering"
    / "SKILL.md"
)
SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]
FENCE = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)
WRAPPED_UTILS = re.compile(r"\b(grep|cat|head|tail|sed|awk|find)\b")


def _text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter() -> dict:
    return yaml.safe_load(_text().split("---", 2)[1])


def test_frontmatter_credits_human_and_has_short_description():
    fm = _frontmatter()
    assert fm["name"] == SKILL_MD.parent.name
    assert len(fm["description"]) <= 60 and fm["description"].endswith(".")
    assert fm["author"].startswith("Yasser Alattas (@yalattas)")
    assert fm["platforms"]


def test_title_and_section_order():
    text = _text()
    assert "\n# Kyverno Policy Engineering Skill\n" in text
    positions = [text.find(f"\n{s}\n") for s in SECTIONS]
    assert -1 not in positions, dict(zip(SECTIONS, positions))
    assert positions == sorted(positions)


def test_prose_names_native_tools_not_shell_utilities():
    prose = FENCE.sub("", _text())
    assert not WRAPPED_UTILS.findall(prose)
    for tool in ("`terminal`", "`read_file`", "`search_files`"):
        assert tool in prose


def test_shell_snippets_do_not_pipe_through_grep():
    for lang, body in FENCE.findall(_text()):
        if lang == "bash":
            assert "| grep" not in body


def test_yaml_examples_use_cel_policy_api():
    yaml_blocks = [body for lang, body in FENCE.findall(_text()) if lang == "yaml"]
    assert yaml_blocks
    for body in yaml_blocks:
        doc = yaml.safe_load(body)
        assert doc["apiVersion"] == "policies.kyverno.io/v1"
        assert doc["kind"] not in {"Policy", "ClusterPolicy"}
        assert doc["spec"]["validationActions"] == ["Audit"]
