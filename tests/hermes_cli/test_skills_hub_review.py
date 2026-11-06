"""Blocked hub packages need an explicit local review of the exact installed bytes."""

import argparse
from io import StringIO
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
from rich.console import Console

from hermes_cli import skills_hub as cli_hub
from hermes_cli.subcommands.skills import build_skills_parser
from tools.skills_hub import HubLockFile
from tools.skills_hub_models import SkillBundle, SkillMeta


@pytest.fixture
def review_install(monkeypatch, tmp_path):
    home = tmp_path / "review-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    identifier = "example/repo/setup-fixture"
    bundle = SkillBundle(
        "setup-fixture",
        {"SKILL.md": "---\nname: setup-fixture\ndescription: Local setup fixture.\n---\nRead references/devices.md before preparing a drive.\n",
         "references/devices.md": "Manual formatting example, never executed during installation: `sudo mkfs.ext4 /dev/EXAMPLE`.\n",
         "assets/model.bin": b"model fixture\x00\x01"},
        "github", identifier, "community",
    )
    meta = SkillMeta(bundle.name, "Local setup fixture.", "github", identifier, "community")
    source = SimpleNamespace(source_id=lambda: "github", inspect=lambda _id: meta, fetch=lambda _id: bundle)
    monkeypatch.setattr(cli_hub, "_sources", lambda: [source])
    monkeypatch.setattr(cli_hub, "_record_skill_install", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_hub, "_print_tier1_advisory", lambda *_a: None)
    monkeypatch.setattr(cli_hub, "_finish_change", lambda *_a: None)
    monkeypatch.setattr(cli_hub, "_announce_blueprint", lambda *_a: None)
    sink = StringIO()
    console = Console(file=sink, force_terminal=True, color_system=None, width=180)
    monkeypatch.setattr(cli_hub, "_console", console)
    parser = argparse.ArgumentParser()
    build_skills_parser(parser.add_subparsers(), cmd_skills=cli_hub.skills_command)

    def install(*flags):
        args = parser.parse_args(["skills", "install", identifier, *flags])
        return cli_hub.skills_command(args)

    return SimpleNamespace(home=home, bundle=bundle, sink=sink, console=console, install=install)


@pytest.mark.parametrize("flags,tty,answer,installed", [
    ([], True, "approve", False),
    (["--force"], True, "approve", False),
    (["--review"], False, "approve", False),
    (["--review", "--yes"], True, "approve", False),
    (["--review"], True, "yes", False),
    (["--review"], True, "approve", True),
])
def test_only_explicit_local_hash_review_installs(review_install, monkeypatch, flags, tty, answer, installed):
    fixture = review_install
    monkeypatch.setattr(cli_hub.sys.stdin, "isatty", lambda: tty)

    def respond(prompt):
        if "full package hash" not in prompt:
            return "yes"
        displayed = re.search(r"Exact package: (sha256:[0-9a-f]{64})", fixture.sink.getvalue())
        assert displayed is not None
        return f"approve {displayed[1]}" if answer == "approve" else answer

    monkeypatch.setattr(fixture.console, "input", respond)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    result = fixture.install(*flags)
    target = fixture.home / "skills" / fixture.bundle.name
    assert target.exists() is installed
    assert (result == 1) is not installed
    if not installed:
        assert HubLockFile().get_installed(fixture.bundle.name) is None
        return
    entry = HubLockFile().get_installed(fixture.bundle.name)
    review = entry["scan_provenance"]["operator_review"]
    assert review["bundle_hash"] == entry["scan_provenance"]["bundle_hash"]
    assert entry["scan_verdict"] == "dangerous"  # review never rewrites the scanner's verdict
    assert (target / "assets/model.bin").read_bytes() == fixture.bundle.files["assets/model.bin"]
    audit = (fixture.home / "skills/.hub/audit.log").read_text()
    assert "REVIEW_APPROVED" in audit and review["bundle_hash"] in audit
    # A new version gets no inherited approval, even with --force.
    fixture.bundle.files["assets/model.bin"] = b"changed model"
    assert fixture.install("--force") == 1
    assert (target / "assets/model.bin").read_bytes() == b"model fixture\x00\x01"


@pytest.mark.parametrize("change_at", ["review", "install"])
def test_changes_during_either_confirmation_cannot_be_installed(review_install, monkeypatch, change_at):
    fixture = review_install
    monkeypatch.setattr(cli_hub.sys.stdin, "isatty", lambda: True)
    pending = fixture.home / "skills/.hub/quarantine" / fixture.bundle.name

    def mutate():
        (pending / "assets/model.bin").write_bytes(b"changed after scanning")

    def approve(_prompt):
        displayed = re.search(r"Exact package: (sha256:[0-9a-f]{64})", fixture.sink.getvalue())
        assert displayed is not None
        if change_at == "review":
            mutate()
        return f"approve {displayed[1]}"

    def confirm(_prompt):
        if change_at == "install":
            mutate()
        return "yes"

    monkeypatch.setattr(fixture.console, "input", approve)
    monkeypatch.setattr("builtins.input", confirm)
    assert fixture.install("--review") == 1
    assert not (fixture.home / "skills" / fixture.bundle.name).exists()
    assert HubLockFile().get_installed(fixture.bundle.name) is None
    assert "Package changed after operator review" in fixture.sink.getvalue()
