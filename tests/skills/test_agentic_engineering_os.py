from pathlib import Path
import subprocess

from agent.skill_utils import parse_frontmatter, skill_matches_platform


ROOT = Path(__file__).resolve().parents[2]
AGENTIC_SKILLS = ("code-structure", "review-loop", "simplify-code")
DOC_SLUG = "developer-guide/agentic-engineering-os"


def _skill_path(name: str) -> Path:
    return ROOT / "skills" / "software-development" / name / "SKILL.md"


def _load_skill(name: str):
    path = _skill_path(name)
    content = path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)
    return path, frontmatter, body


def test_agentic_engineering_skills_parse_with_production_parser():
    seen_names = set()

    for name in AGENTIC_SKILLS:
        path, frontmatter, body = _load_skill(name)

        assert path.exists()
        assert frontmatter["name"] == name
        assert frontmatter["description"]
        assert len(frontmatter["description"]) <= 180
        assert frontmatter["version"]
        assert frontmatter["author"]
        assert frontmatter["license"] == "MIT"
        assert isinstance(frontmatter.get("platforms"), list)
        assert skill_matches_platform(frontmatter)
        assert body.strip()
        assert name not in seen_names
        seen_names.add(name)


def test_new_agentic_engineering_skills_have_clear_routing_boundaries():
    code_structure = _load_skill("code-structure")[1]
    review_loop = _load_skill("review-loop")[1]
    simplify_code = _load_skill("simplify-code")[1]

    assert "review-loop" in code_structure["metadata"]["hermes"]["related_skills"]
    assert "simplify-code" in code_structure["metadata"]["hermes"]["related_skills"]
    assert "requesting-code-review" in review_loop["metadata"]["hermes"]["related_skills"]
    assert "code-structure" in simplify_code["metadata"]["hermes"]["related_skills"]
    assert "review-loop" in simplify_code["metadata"]["hermes"]["related_skills"]


def test_local_references_ignore_rules_allow_only_readme_and_gitkeep():
    assert (ROOT / ".references" / "README.md").exists()
    assert (ROOT / ".references" / ".gitkeep").exists()

    ignored = subprocess.run(
        ["git", "check-ignore", ".references/vendor/docs.md"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert ignored.returncode == 0
    assert ".references/vendor/docs.md" in ignored.stdout

    for tracked_exception in (".references/README.md", ".references/.gitkeep"):
        result = subprocess.run(
            ["git", "check-ignore", tracked_exception],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 1, result.stdout + result.stderr


def test_agentic_engineering_docs_are_wired_into_navigation_and_llms_index():
    doc_path = ROOT / "website" / "docs" / f"{DOC_SLUG}.md"
    assert doc_path.exists()
    doc = doc_path.read_text(encoding="utf-8")
    assert "## Value proposition" in doc
    assert "No core-surface expansion" in doc

    sidebar = (ROOT / "website" / "sidebars.ts").read_text(encoding="utf-8")
    llms_script = (ROOT / "website" / "scripts" / "generate-llms-txt.py").read_text(
        encoding="utf-8"
    )
    assert DOC_SLUG in sidebar
    assert DOC_SLUG in llms_script
