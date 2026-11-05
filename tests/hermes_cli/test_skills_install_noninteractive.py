"""A missing confirmation terminal fails installation with an actionable hint."""

from argparse import ArgumentParser
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from hermes_cli import skills_hub as cli_hub
from hermes_cli.subcommands.skills import build_skills_parser
from tools import skills_hub as hub
from tools.skills_hub_models import SkillBundle


@pytest.mark.parametrize("answer, expected_status, hint", [
    (EOFError, 1, True),
    ("n", 0, False),
])
def test_install_cancellation_exit_status(monkeypatch, tmp_path, answer, expected_status, hint):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    hub_dir = home / "skills" / ".hub"
    for name, value in {
        "SKILLS_DIR": home / "skills", "HUB_DIR": hub_dir,
        "LOCK_FILE": hub_dir / "lock.json", "QUARANTINE_DIR": hub_dir / "quarantine",
        "AUDIT_LOG": hub_dir / "audit.log", "TAPS_FILE": hub_dir / "taps.json",
        "INDEX_CACHE_DIR": hub_dir / "index-cache",
    }.items():
        monkeypatch.setattr(hub, name, value)
    source = SimpleNamespace(source_id=lambda: "github")
    bundle = SkillBundle(
        name="confirmation-fixture", source="github", identifier="org/repo/confirmation-fixture",
        trust_level="community", files={"SKILL.md": "---\nname: confirmation-fixture\n"
                                      "description: Format a simple table.\n---\nUse a Markdown table.\n"})
    monkeypatch.setattr(cli_hub, "_sources", lambda: [source])
    monkeypatch.setattr(cli_hub, "_resolve_source_meta_and_bundle", lambda *a: (None, bundle, source))
    monkeypatch.setattr(cli_hub, "_print_tier1_advisory", lambda *a: None)
    monkeypatch.setattr(cli_hub, "_record_skill_install", lambda *a, **k: None)
    sink = StringIO()
    monkeypatch.setattr(cli_hub, "_console", Console(file=sink, width=120, color_system=None))

    def respond(prompt):
        if answer is EOFError:
            raise EOFError
        return answer

    monkeypatch.setattr("builtins.input", respond)
    parser = ArgumentParser()
    build_skills_parser(parser.add_subparsers(), cmd_skills=cli_hub.skills_command)
    args = parser.parse_args(["skills", "install", bundle.identifier])
    status = args.func(args) or 0

    assert status == expected_status
    assert "Installation cancelled." in sink.getvalue()
    assert ("No terminal to answer on: re-run with --yes to confirm (the security scan still runs)."
            in sink.getvalue()) is hint
    assert not (home / "skills" / bundle.name).exists()
    assert hub.HubLockFile().list_installed() == []
