"""Render catalog entries resolve vendor content and fail closed without it."""

# JSON decodes Hermes tool replies; mocks keep these tests independent of GitHub.
import json
from unittest.mock import Mock

import pytest

# Use the production resolver, package representation, installer, scanner and skill reader.
from tools.skills_guard import scan_skill, should_allow_install
from tools.skills_hub_install import install_from_quarantine, quarantine_bundle
from tools.skills_hub_models import SkillBundle
from tools.skills_hub_official import OptionalSkillSource
from tools.skills_tool import skill_view

# Cover each proposed catalog entry through its real public install identifier.
RENDER_SKILLS = [
    ("devops", "render-background-workers"),
    ("devops", "render-blueprints"),
    ("devops", "render-cli"),
    ("devops", "render-cron-jobs"),
    ("devops", "render-debug"),
    ("devops", "render-deploy"),
    ("devops", "render-disks"),
    ("devops", "render-docker"),
    ("devops", "render-domains"),
    ("devops", "render-env-vars"),
    ("devops", "render-keyvalue"),
    ("mcp", "render-mcp"),
    ("migration", "render-migrate-from-heroku"),
    ("devops", "render-monitor"),
    ("devops", "render-networking"),
    ("devops", "render-postgres"),
    ("devops", "render-private-services"),
    ("devops", "render-scaling"),
    ("devops", "render-static-sites"),
    ("devops", "render-web-services"),
    ("devops", "render-workflows"),
]


@pytest.mark.parametrize(("category", "name"), RENDER_SKILLS)
def test_render_entry_installs_upstream_instructions_and_references(category, name):
    """An official listing must deliver Render's full bundle with normal scanning."""
    source = OptionalSkillSource()
    identifier = f"official/{category}/{name}"
    # The small downloaded fixture includes a reference to detect incomplete installs.
    upstream = SkillBundle(
        name=name,
        files={
            "SKILL.md": f"---\nname: {name}\ndescription: Check Render resources.\n---\n"
                        "# Render task\nRead references/access.md before inspecting a workspace.\n",
            "references/access.md": "Use the connected Render tools to verify the workspace.",
        },
        source="github",
        identifier=f"render-oss/skills/skills/{name}",
        trust_level="community",
        metadata={"source_revision": "a" * 40},
    )
    # Only GitHub is replaced: the catalog lookup and downstream install remain real.
    source._github = Mock()
    source._github.fetch.return_value = upstream
    bundle = source.fetch(identifier)
    source._github.fetch.assert_called_once_with(f"render-oss/skills/skills/{name}")
    assert bundle is not None
    assert bundle.source == "official"
    assert bundle.trust_level == "trusted"  # Live vendor content must still pass a scan.
    assert bundle.metadata["source_revision"] == upstream.metadata["source_revision"]

    # Stage, scan and install using the test runner's isolated Hermes profile.
    staged = quarantine_bundle(bundle)
    scan = scan_skill(staged, source=bundle.source)
    assert should_allow_install(scan)[0]
    install_from_quarantine(staged, name, "", bundle, scan)

    # The agent must be able to load the downloaded instructions and their reference.
    instructions = json.loads(skill_view(name, preprocess=False))
    reference = json.loads(skill_view(name, file_path="references/access.md", preprocess=False))
    assert instructions["success"] and "references/access.md" in instructions["content"]
    assert reference["success"] and reference["content"] == upstream.files["references/access.md"]


@pytest.mark.parametrize(("category", "name"), RENDER_SKILLS)
def test_render_entry_does_not_install_the_catalog_description_when_download_fails(category, name):
    """A network failure cannot silently replace operational instructions with a listing."""
    source = OptionalSkillSource()
    source._github = Mock()
    source._github.fetch.return_value = None
    assert source.fetch(f"official/{category}/{name}") is None
