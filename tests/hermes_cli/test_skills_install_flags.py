"""
Tests for --yes / --force flag separation in `hermes skills install`.

--yes / -y  → skip_confirm (bypass interactive prompt, needed in TUI mode)
--force     → force (install despite blocked scan verdict)

Based on PR #1595 by 333Alden333 (salvaged).
"""

import sys
from types import SimpleNamespace


def test_cli_skills_install_yes_sets_skip_confirm(monkeypatch):
    """--yes should set skip_confirm=True but NOT force."""
    from hermes_cli.main import main

    captured = {}

    def fake_skills_command(args):
        captured["identifier"] = args.identifier
        captured["force"] = args.force
        captured["yes"] = args.yes

    monkeypatch.setattr("hermes_cli.skills_hub.skills_command", fake_skills_command)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "skills", "install", "official/email/agentmail", "--yes"],
    )

    main()

    assert captured["identifier"] == "official/email/agentmail"
    assert captured["yes"] is True
    assert captured["force"] is False


def test_cli_skills_install_y_alias(monkeypatch):
    """-y should behave the same as --yes."""
    from hermes_cli.main import main

    captured = {}

    def fake_skills_command(args):
        captured["yes"] = args.yes
        captured["force"] = args.force

    monkeypatch.setattr("hermes_cli.skills_hub.skills_command", fake_skills_command)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "skills", "install", "test/skill", "-y"],
    )

    main()

    assert captured["yes"] is True
    assert captured["force"] is False


def test_cli_skills_install_force_sets_force(monkeypatch):
    """--force should set force=True but NOT yes."""
    from hermes_cli.main import main

    captured = {}

    def fake_skills_command(args):
        captured["force"] = args.force
        captured["yes"] = args.yes

    monkeypatch.setattr("hermes_cli.skills_hub.skills_command", fake_skills_command)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "skills", "install", "test/skill", "--force"],
    )

    main()

    assert captured["force"] is True
    assert captured["yes"] is False


def test_cli_skills_install_force_and_yes_together(monkeypatch):
    """--force --yes should set both flags."""
    from hermes_cli.main import main

    captured = {}

    def fake_skills_command(args):
        captured["force"] = args.force
        captured["yes"] = args.yes

    monkeypatch.setattr("hermes_cli.skills_hub.skills_command", fake_skills_command)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "skills", "install", "test/skill", "--force", "--yes"],
    )

    main()

    assert captured["force"] is True
    assert captured["yes"] is True


def test_cli_skills_install_no_flags(monkeypatch):
    """Without flags, both force and yes should be False."""
    from hermes_cli.main import main

    captured = {}

    def fake_skills_command(args):
        captured["force"] = args.force
        captured["yes"] = args.yes

    monkeypatch.setattr("hermes_cli.skills_hub.skills_command", fake_skills_command)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "skills", "install", "test/skill"],
    )

    main()

    assert captured["force"] is False
    assert captured["yes"] is False
