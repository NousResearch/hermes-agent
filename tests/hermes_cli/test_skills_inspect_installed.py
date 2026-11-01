"""``hermes skills inspect`` resolves installed skills before the hub registries (#25087).

Inspection answered from hub registries only, so a local skill that ``hermes skills list`` shows and
the runtime resolves printed "No skill named ... found in any source" — and exited 0, so agent-driven
install verification read a working install as a failed one.
"""

from argparse import Namespace
from io import StringIO

import pytest
from rich.console import Console

import hermes_cli.skills_hub as cli_hub


@pytest.fixture
def installed_skill(tmp_path, monkeypatch):
    from tools import skills_tool

    skills_dir = tmp_path / "hermes" / "skills"
    skill_dir = skills_dir / "qa-probe"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: qa-probe\ndescription: Local probe skill\ntags: [probe]\n---\n"
        "\n# QA Probe\n\nBody line.\n", encoding="utf-8")
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    skills_tool.clear_skills_cache()
    try:
        yield skill_dir
    finally:
        skills_tool.clear_skills_cache()


def test_inspect_reports_an_installed_skill(installed_skill, monkeypatch):
    monkeypatch.setattr(cli_hub, "_sources",
                        lambda: pytest.fail("inspect queried the hub for an installed skill"))
    console = Console(file=StringIO(), width=100)

    assert cli_hub.do_inspect("qa-probe", console=console) is None

    out = console.file.getvalue()
    assert "Description: Local probe skill" in out
    assert "Source: local" in out
    assert "Body line." in out

    info = cli_hub.inspect_skill("qa-probe")
    assert info["name"] == "qa-probe"
    assert info["description"] == "Local probe skill"
    assert info["source"] == "local"
    assert info["tags"] == ["probe"]
    assert "Body line." in info["skill_md_preview"]


def test_unresolved_inspect_exits_non_zero(monkeypatch, capsys):
    monkeypatch.setattr(cli_hub, "_sources", list)

    code = cli_hub.skills_command(
        Namespace(skills_action="inspect", identifier="no-such-skill-anywhere"))

    assert code == 1
    assert "No skill named 'no-such-skill-anywhere' found in any source." in capsys.readouterr().out
