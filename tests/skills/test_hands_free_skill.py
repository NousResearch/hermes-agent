"""Exercise the shipped Hands Free bundle through the real optional catalog and loader.

These are packaging/loading contracts, not claims that an LLM obeys the prose.
Only external fetch and dependency-install boundaries are blocked; discovery,
frontmatter parsing, readiness and reference resolution use production code.
"""

import json
import shutil
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_REL = Path("autonomous-ai-agents") / "hands-free"
SKILL_SOURCE = REPO_ROOT / "optional-skills" / SKILL_REL
IDENTIFIER = "official/autonomous-ai-agents/hands-free"
REFERENCES = ("references/faceless-tutorial.md", "references/red-team-gate.md")


@pytest.fixture
def skill_runtime(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    # Keep project discovery and external skill roots out of this integration.
    (home / "config.yaml").write_text(
        "skills:\n  project_discovery: false\n  external_dirs: []\n",
        encoding="utf-8",
    )
    optional_root = tmp_path / "optional-skills"
    shutil.copytree(SKILL_SOURCE, optional_root / SKILL_REL)
    monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", str(optional_root))

    # Import after isolation, so even import-time paths belong to the test home.
    import pm
    from hermes_constants import get_hermes_home
    from tools import skills_tool
    from tools.skills_hub_official import OptionalSkillSource

    def forbidden_side_effect(*args, **kwargs):
        pytest.fail("This local Markdown skill must not fetch or install dependencies")

    monkeypatch.setattr(OptionalSkillSource, "_get_github", forbidden_side_effect)
    monkeypatch.setattr(pm, "ensure", forbidden_side_effect)
    assert get_hermes_home() == home
    skills_tool.clear_skills_cache()
    try:
        yield home, OptionalSkillSource(), skills_tool
    finally:
        skills_tool.clear_skills_cache()


@pytest.fixture
def installed_skill(skill_runtime):
    home, source, loader = skill_runtime
    bundle = source.fetch(IDENTIFIER)
    assert bundle is not None, "The shipped optional skill must be fetchable offline"
    destination = home / "skills" / SKILL_REL
    # Materialize the actual fetched payload, not a fabricated SKILL.md. This
    # tests the install layout without invoking the CLI's installation/setup UX.
    for relative, content in bundle.files.items():
        target = (destination / relative).resolve()
        assert target.is_relative_to(destination.resolve()), relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return destination, loader


def test_optional_catalog_fetches_complete_bundle_without_activating_it(skill_runtime):
    home, source, loader = skill_runtime
    matches = [entry for entry in source.list_local() if entry.identifier == IDENTIFIER]
    assert len(matches) == 1
    assert matches[0].name == "hands-free"

    bundle = source.fetch(matches[0].identifier)
    assert bundle is not None
    assert bundle.name == matches[0].name
    # Normalize separators only: local fetch uses native paths on Windows.
    payload = {Path(name).as_posix(): content for name, content in bundle.files.items()}
    expected = {
        path.relative_to(SKILL_SOURCE).as_posix(): path.read_bytes()
        for path in SKILL_SOURCE.rglob("*")
        if path.is_file()
    }
    assert {"SKILL.md", *REFERENCES} <= payload.keys()
    assert payload == expected
    assert not (home / "skills" / SKILL_REL).exists()
    assert not any(row["name"] == "hands-free" for row in loader.find_all_skills())
    unavailable = json.loads(loader.skill_view("hands-free", preprocess=False))
    assert unavailable["success"] is False
    assert "content" not in unavailable


def test_installed_skill_is_discoverable_and_ready_without_setup(installed_skill):
    destination, loader = installed_skill
    entries = [row for row in loader.find_all_skills() if row["name"] == "hands-free"]
    assert len(entries) == 1
    assert entries[0]["category"] == "autonomous-ai-agents"

    loaded = json.loads(loader.skill_view(entries[0]["name"], preprocess=False))
    assert loaded["success"] is True, loaded
    assert Path(loaded["skill_dir"]) == destination
    assert loaded["description"] == entries[0]["description"]
    assert loaded["content"] == (SKILL_SOURCE / "SKILL.md").read_text(encoding="utf-8")
    assert loaded["readiness_status"] == "available"
    assert loaded["setup_needed"] is False
    assert loaded["required_environment_variables"] == []
    assert loaded["missing_credential_files"] == []
    assert "deps_note" not in loaded
    assert set(REFERENCES) <= set(loaded["linked_files"]["references"])


@pytest.mark.parametrize("reference", REFERENCES)
def test_advertised_reference_loads_complete_shipped_content(installed_skill, reference):
    destination, loader = installed_skill
    loaded = json.loads(loader.skill_view("hands-free", preprocess=False))
    assert loaded["success"] is True, loaded
    assert reference in loaded["linked_files"]["references"]
    result = json.loads(loader.skill_view("hands-free", file_path=reference))
    assert result["success"] is True, result
    assert result["file"] == reference
    assert result["content"] == (SKILL_SOURCE / reference).read_text(encoding="utf-8")
    assert (destination / reference).is_file()
    # Supporting Markdown is not a standalone skill offered to the model.
    assert not any(row["name"] == Path(reference).stem for row in loader.find_all_skills())


def test_reference_lookup_refuses_to_read_outside_installed_skill(installed_skill):
    destination, loader = installed_skill
    outside = destination.parent / "private-evidence.md"
    outside.write_text("Private evidence outside the skill bundle.\n", encoding="utf-8")
    for escaped_path in ("../private-evidence.md", str(outside.resolve())):
        result = json.loads(loader.skill_view("hands-free", file_path=escaped_path))
        assert result["success"] is False, result
        assert "content" not in result
    # A denied request must not prevent subsequent valid reference access.
    allowed = json.loads(loader.skill_view("hands-free", file_path=REFERENCES[0]))
    assert allowed["success"] is True, allowed
    assert allowed["content"] == (SKILL_SOURCE / REFERENCES[0]).read_text(encoding="utf-8")
