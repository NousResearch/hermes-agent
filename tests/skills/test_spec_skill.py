"""Behavior contracts for the bundled /spec intake skill."""

from pathlib import Path
from unittest.mock import patch

import yaml

from agent.skill_commands import build_skill_invocation_message, scan_skill_commands
from tools.skill_manager_tool import _validate_frontmatter


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / "skills"
SKILL_MD = SKILLS_ROOT / "productivity" / "spec" / "SKILL.md"


def _content() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter(content: str) -> dict:
    _, raw, _ = content.split("---", 2)
    return yaml.safe_load(raw)


def test_spec_skill_has_valid_v2_frontmatter() -> None:
    content = _content()

    assert _validate_frontmatter(content) is None
    frontmatter = _frontmatter(content)
    assert frontmatter["name"] == "spec"
    assert frontmatter["version"] == "2.0.0"
    assert "structured interview" in frontmatter["description"].lower()


def test_spec_skill_registers_as_slash_command_and_loads_instruction() -> None:
    instruction = "DRY RUN: add a troubleshooting sentence to README.md"

    with (
        patch("tools.skills_tool.SKILLS_DIR", SKILLS_ROOT),
        patch("tools.skills_tool._get_disabled_skill_names", return_value=set()),
        patch("agent.skill_utils.get_external_skills_dirs", return_value=[]),
    ):
        commands = scan_skill_commands()
        message = build_skill_invocation_message("/spec", instruction)

    assert commands["/spec"]["name"] == "spec"
    assert Path(commands["/spec"]["skill_md_path"]).resolve() == SKILL_MD.resolve()
    assert message is not None
    assert "user has invoked the \"spec\" skill" in message
    assert instruction in message
    assert "DRY RUN — NO LINEAR MUTATION" in message


def test_spec_skill_bridge_contract_is_complete_and_fail_closed() -> None:
    content = _content()

    # Immediate flow requires all three bridge selectors, not merely a copied
    # routing name. Gated packets must be explicitly held outside unstarted.
    for required in (
        "Build Ops / `BUI`",
        "agent:<profile>",
        "hermes profile list",
        "`enabled` is `true`",
        "`dry_run` is `false`",
        "routing_label_prefix",
        "status_types",
        "allowed_profiles",
        "issue_id_allowlist",
        "max_creates_per_tick",
        "currently `unstarted`",
        "backlog/triage by default",
    ):
        assert required in content

    assert "Never create a missing label" in content
    assert "ROUTING REQUESTED:" in content
    assert "label/profile/bridge mapping unverified" in content
    assert "never describe it as queued for the bridge" in content


def test_spec_skill_enforces_one_issue_approval_and_machine_readable_risk() -> None:
    content = _content()

    assert "Write one Linear issue at the end" in content
    assert "operator gives explicit approval" in content
    assert "Create exactly one issue" in content
    assert "Do not build, dispatch, route live, or create a Kanban card" in content

    for flag in (
        "auth",
        "secrets",
        "payments",
        "migration",
        "production",
        "legal",
        "trading",
        "security",
        "external-communication",
    ):
        assert f"    {flag}: false" in content


def test_spec_skill_dry_run_forbids_mutation_and_dispatch() -> None:
    content = _content()

    assert "DRY RUN — NO LINEAR MUTATION" in content
    for forbidden_action in (
        "issueCreate",
        "issueUpdate",
        "Kanban mutation tools",
        "Git mutation commands",
        "build/deploy commands",
    ):
        assert forbidden_action in content
    assert "never creates or dispatches work" in content
