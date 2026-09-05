"""Offline tests for the optional bo2bot-messaging skill."""
import ast
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "messaging" / "bo2bot-messaging"
SKILL_PATH = SKILL_DIR / "SKILL.md"

BUNDLED = [
    "references/Bo2bot_For_LLMs.md",
    "references/Bo2bot_Hermes_Kickoff.md",
    "references/bo2bot.env.sample",
    "references/credentials-setup.md",
    "scripts/bo2bot_cred_manager.py",
    "scripts/bo2bot_loader.py",
    "scripts/bo2bot-login.sh",
    "scripts/bo2bot-setup.sh",
    "scripts/bo2bot-validate.sh",
]


def _frontmatter_and_body():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---")
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "frontmatter must close with ---"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    body = content[m.end() + 3 :]
    return fm, body


def test_skill_file_exists():
    assert SKILL_PATH.is_file()


def test_frontmatter_required_fields():
    fm, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "bo2bot-messaging"
    assert fm["name"] == SKILL_DIR.name
    hermes = fm["metadata"]["hermes"]
    assert hermes["tags"]
    assert "related_skills" in hermes


def test_description_hardline():
    fm, _ = _frontmatter_and_body()
    desc = fm["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars; hardline is 60"
    assert desc.endswith(".")
    assert not re.search(
        r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b",
        desc,
        re.I,
    )


def test_author_credits_human_first():
    fm, _ = _frontmatter_and_body()
    assert "Abhijeet Kushwaha" in fm["author"]
    assert "@bo2bot" in fm["author"]
    assert not str(fm["author"]).startswith("Hermes Agent")


def test_required_credential_files():
    fm, _ = _frontmatter_and_body()
    files = fm.get("required_credential_files") or []
    assert files, "expected required_credential_files"
    paths = {entry["path"] for entry in files}
    assert "secrets/bo2bot.env" in paths


def test_body_structure_and_size():
    _, body = _frontmatter_and_body()
    for section in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert section in body, f"missing section: {section}"
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert len(content) <= 100_000
    assert content.count("\n") <= 350, "prefer a lean official SKILL.md"


def test_bundled_files_exist():
    _, body = _frontmatter_and_body()
    for rel in BUNDLED:
        assert (SKILL_DIR / rel).is_file(), f"missing bundled file: {rel}"
        assert rel in body or rel.split("/")[-1] in body


def test_sample_env_has_required_keys_only():
    sample = (SKILL_DIR / "references" / "bo2bot.env.sample").read_text(encoding="utf-8")
    for key in (
        "BO2BOT_ACCOUNT_ID",
        "BO2BOT_HANDLE",
        "BO2BOT_PUBLIC_ADDRESS",
        "BO2BOT_AUTH_KEY",
    ):
        assert f"{key}=" in sample
    assert "REPLACE" in sample or "yourhandle" in sample
    assert "bo2bot_" in sample.lower()


def test_python_helpers_parse():
    for name in ("bo2bot_cred_manager.py", "bo2bot_loader.py"):
        src = (SKILL_DIR / "scripts" / name).read_text(encoding="utf-8")
        ast.parse(src)


def test_no_machine_local_paths():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert not re.search(r"/home/(?!runner\b)[a-z0-9_-]+/", content)
    assert not re.search(r"[A-Z]:\\+Users\\+", content)


def test_mentions_api_not_mcp_credentials():
    content = SKILL_PATH.read_text(encoding="utf-8").lower()
    assert "api" in content
    assert "mcp" in content  # explicit exclusion guidance
    assert "not mcp" in content or "never mcp" in content


def test_procedure_steps_have_done_when():
    _, body = _frontmatter_and_body()
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## )", body, re.MULTILINE | re.DOTALL)
    assert len(steps) >= 4
    for step in steps:
        assert "Done when" in step, f"step missing completion criterion: {step[:80]!r}"
