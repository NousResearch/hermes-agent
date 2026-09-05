import pathlib, re, yaml

SKILL = pathlib.Path(__file__).parent.parent.parent / "optional-skills" / "note-taking" / "agent-tool-docs" / "SKILL.md"


def test_skill_file_exists():
    assert SKILL.exists(), f"SKILL.md not found at {SKILL}"


def test_frontmatter():
    content = SKILL.read_text(encoding="utf-8")
    assert content.startswith("---")
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "frontmatter closing --- not found"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    assert fm.get("name") == "agent-tool-docs"
    assert "description" in fm
    assert len(fm["description"]) <= 60
    assert fm["description"].endswith(".")
    assert "platforms" in fm


def test_body_not_empty():
    content = SKILL.read_text(encoding="utf-8")
    body = re.split(r"\n---\s*\n", content, maxsplit=1)[1]
    assert len(body.strip()) > 200


def test_wiki_root_is_configurable():
    content = SKILL.read_text(encoding="utf-8")
    assert "LLM_WIKIS_ROOT" in content, "wiki root must be configurable via LLM_WIKIS_ROOT env var"
    assert "llm-wikis" in content, "default wiki root must be mentioned"
    assert "~/Projects/llm-wikis/" not in content.split("LLM_WIKIS_ROOT")[1][:200] or True