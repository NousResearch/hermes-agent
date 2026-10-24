"""Hermetic tests for the shieldnode skill.

The skill ships no scripts: it is a document contract telling the agent which
endpoints to call and how to behave. These tests enforce that contract, so a
future edit cannot silently break the parts a reader copies and pastes.

Stdlib + pytest only. No network, no API keys, no filesystem writes.

    scripts/run_tests.sh tests/skills/test_shieldnode_skill.py -q
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_REL = ("optional-skills", "security", "shieldnode")
_CANDIDATES = [
    _HERE.parent.parent.parent.joinpath(*_REL),  # hermes-agent (tests/skills/)
    _HERE.parent.parent.joinpath(*_REL),         # standalone layout
]
SKILL_DIR = next((c for c in _CANDIDATES if (c / "SKILL.md").exists()), _CANDIDATES[0])
SKILL_MD = SKILL_DIR / "SKILL.md"

# The one place the skill is allowed to tell the agent to persist service docs.
# Duplicating this path in a second file is what drifted before, so the test
# asserts exactly one file states it.
SERVICE_DOC_PATH = "~/.hermes/shieldnode/services/<slug>.md"

MANDATED_SECTIONS = [
    "When to Use",
    "Prerequisites",
    "How to Run",
    "Quick Reference",
    "Procedure",
    "Pitfalls",
    "Verification",
]

MARKETING_WORDS = ("powerful", "comprehensive", "seamless", "advanced", "revolutionary")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _frontmatter(text: str) -> dict[str, str]:
    """Parse the top-level scalar keys of the frontmatter without PyYAML."""
    assert text.startswith("---\n"), "SKILL.md must open with a frontmatter block"
    end = text.index("\n---\n", 3)
    out: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line or line.startswith((" ", "\t", "#")):
            continue  # nested mapping (metadata:) or comment
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def _fenced_blocks(text: str, lang: str) -> list[str]:
    return re.findall(rf"^```{lang}\n(.*?)^```", text, re.MULTILINE | re.DOTALL)


@pytest.fixture(scope="module")
def skill_text() -> str:
    return _read(SKILL_MD)


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict[str, str]:
    return _frontmatter(skill_text)


def test_skill_directory_layout():
    assert SKILL_MD.exists(), f"SKILL.md not found under {SKILL_DIR}"
    assert (SKILL_DIR / "references").is_dir()
    # Scripts belong in scripts/; this skill intentionally ships none.
    assert not (SKILL_DIR / "scripts").exists()


def test_frontmatter_has_required_keys(frontmatter):
    for key in ("name", "description", "version", "author", "license"):
        assert key in frontmatter, f"missing frontmatter key: {key}"
    assert frontmatter["name"] == "shieldnode"


def test_description_meets_authoring_standard(frontmatter):
    description = frontmatter["description"]
    assert len(description) <= 60, f"description is {len(description)} chars, max 60"
    assert description.endswith("."), "description must end with a period"
    assert description.count(".") == 1, "description must be a single sentence"
    lowered = description.lower()
    for word in MARKETING_WORDS:
        assert word not in lowered, f"marketing word in description: {word}"
    assert "shieldnode" not in lowered, "description must not repeat the skill name"


def test_author_credits_a_human_with_handle(frontmatter):
    # "Real Name (github-handle)", human first per the authoring standards.
    assert re.match(r"^[^(]+\([\w-]+\)", frontmatter["author"]), frontmatter["author"]


def test_title_and_section_order(skill_text):
    headings = re.findall(r"^#{1,2} (.+)$", skill_text, re.MULTILINE)
    assert headings[0] == "ShieldNode Skill"
    found = [h for h in headings if h in MANDATED_SECTIONS]
    assert found == MANDATED_SECTIONS, f"section order is {found}"


def test_every_referenced_file_exists(skill_text):
    referenced = set(re.findall(r"references/([\w-]+\.md)", skill_text))
    assert referenced, "SKILL.md should point at its references"
    for name in sorted(referenced):
        assert (SKILL_DIR / "references" / name).exists(), f"missing references/{name}"


def test_service_doc_path_is_stated_exactly_once(skill_text):
    """Regression guard: the path used to be duplicated and the copies drifted."""
    assert SERVICE_DOC_PATH in skill_text, "SKILL.md must state the service-doc path"

    others = [
        p for p in sorted((SKILL_DIR / "references").glob("*.md"))
        if re.search(r"services/<[\w-]*slug>\.md", _read(p))
    ]
    assert not others, (
        "the service-doc path must live only in SKILL.md, found a second copy in: "
        + ", ".join(p.name for p in others)
    )


def test_no_conflicting_service_doc_path(skill_text):
    variants = set(re.findall(r"[\w~./]*services/<[\w-]*slug>\.md", skill_text))
    assert variants == {SERVICE_DOC_PATH}, f"conflicting paths in SKILL.md: {variants}"


def test_json_examples_parse(skill_text):
    blocks = _fenced_blocks(skill_text, "json")
    assert blocks, "SKILL.md should document at least one response shape"
    for block in blocks:
        json.loads(block)  # raises on a malformed example


def _unsendable_json_payloads(source: str) -> list[str]:
    """Return a description of every literal `json=` payload that cannot be sent.

    Payloads referencing variables are skipped: nothing can be verified about
    them statically. Literal ones must survive json.dumps, which is what
    catches `[...]` (Ellipsis) and other non-serialisable values.
    """
    problems: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "json":
                continue
            try:
                payload = ast.literal_eval(kw.value)
            except ValueError:
                continue  # references a variable, not a literal
            try:
                json.dumps(payload)
            except TypeError as exc:
                problems.append(f"line {kw.value.lineno}: {exc}")
    return problems


def test_python_recipe_is_valid_and_serialisable():
    """Regression guard: the recipe once passed `messages: [...]`, i.e. Ellipsis,
    which is not JSON-serialisable, so the snippet failed before reaching the
    proxy. Parse it and prove every literal json= payload can actually be sent."""
    recipe = SKILL_DIR / "references" / "approval-recipe.md"
    blocks = _fenced_blocks(_read(recipe), "python")
    assert blocks, "approval-recipe.md must contain the implementation"

    for block in blocks:
        ast.parse(block)  # raises SyntaxError on broken code
        assert not _unsendable_json_payloads(block), _unsendable_json_payloads(block)


def test_the_serialisability_guard_actually_catches_the_old_bug():
    """Prove the guard above is not vacuous by running it on the original bug."""
    bad = 'requests.post(url, json={"model": "x", "messages": [...]})'
    assert _unsendable_json_payloads(bad), "the guard would not catch Ellipsis"

    good = 'requests.post(url, json={"model": "x", "messages": [{"role": "user"}]})'
    assert not _unsendable_json_payloads(good)


def test_no_stale_or_literal_credentials():
    """No obsolete key prefix, and no value that looks like a usable key."""
    for path in [SKILL_MD, *sorted((SKILL_DIR / "references").glob("*.md"))]:
        text = _read(path)
        assert "sk_live_" not in text, f"obsolete key prefix in {path.name}"
        for match in re.findall(r"shieldnode_(?:config_)?([A-Za-z0-9_-]*)", text):
            assert len(match) < 20, f"{path.name} may embed a real key value"


def test_reserved_namespace_endpoints_are_documented(skill_text):
    """The endpoints the skill drives must stay under the reserved namespace,
    which the proxy answers itself and never forwards upstream."""
    for endpoint in ("whoami", "schedule-request", "config-request"):
        assert f"/_shieldnode/{endpoint}" in skill_text, f"undocumented: {endpoint}"

    stray = re.findall(r"proxy\.shieldnode\.app/_(?!shieldnode/)[\w-]+", skill_text)
    assert not stray, f"endpoints outside the reserved namespace: {stray}"


def test_agent_identifies_itself_in_examples(skill_text):
    """X-Agent-Name is what makes an approval prompt name the requester, so the
    examples must model sending it."""
    assert "X-Agent-Name: Hermes" in skill_text
