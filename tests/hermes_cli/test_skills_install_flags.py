"""
Tests for --yes / --force flag separation in `hermes skills install`.

--yes / -y  → skip_confirm (bypass interactive prompt, needed in TUI mode)
--force     → force (install despite blocked scan verdict)

Based on PR #1595 by 333Alden333 (salvaged).
"""

import sys

import pytest


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


def test_cli_skills_install_failure_exits_nonzero(monkeypatch):
    """A failed install must propagate through the top-level CLI exit code."""
    from hermes_cli.main import main

    monkeypatch.setattr("hermes_cli.skills_hub.skills_command", lambda _args: False)
    monkeypatch.setattr(sys, "argv", ["hermes", "skills", "install", "missing/skill", "--yes"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1


def test_cli_skills_install_success_does_not_exit_nonzero(monkeypatch):
    """A successful install keeps the existing zero-exit behavior."""
    from hermes_cli.main import main

    monkeypatch.setattr("hermes_cli.skills_hub.skills_command", lambda _args: True)
    monkeypatch.setattr(sys, "argv", ["hermes", "skills", "install", "valid/skill", "--yes"])

    main()
