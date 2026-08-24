"""
Tests for --yes / -y flag in `hermes skills uninstall` CLI subcommand.

Verifies the parser registers the flag and the value reaches
``do_uninstall(skip_confirm=True)`` through the real ``skills_command``
dispatch path (mirrors ``test_skills_install_flags.py``).
"""

import sys


def _run_uninstall_cli(monkeypatch, argv):
    captured = {}

    def fake_do_uninstall(name, *, skip_confirm=False, **kwargs):
        captured["name"] = name
        captured["skip_confirm"] = skip_confirm

    monkeypatch.setattr("hermes_cli.skills_hub.do_uninstall", fake_do_uninstall)
    monkeypatch.setattr(sys, "argv", argv)

    from hermes_cli.main import main

    main()
    return captured


def test_cli_skills_uninstall_yes_sets_skip_confirm(monkeypatch):
    """`--yes` should propagate to do_uninstall(skip_confirm=True)."""
    captured = _run_uninstall_cli(
        monkeypatch,
        ["hermes", "skills", "uninstall", "test-skill", "--yes"],
    )

    assert captured["name"] == "test-skill"
    assert captured["skip_confirm"] is True


def test_cli_skills_uninstall_y_alias_sets_skip_confirm(monkeypatch):
    """`-y` should behave the same as `--yes`."""
    captured = _run_uninstall_cli(
        monkeypatch,
        ["hermes", "skills", "uninstall", "test-skill", "-y"],
    )

    assert captured["name"] == "test-skill"
    assert captured["skip_confirm"] is True


def test_cli_skills_uninstall_no_flags_keeps_prompt(monkeypatch):
    """Without --yes, skip_confirm must be False (prompt preserved)."""
    captured = _run_uninstall_cli(
        monkeypatch,
        ["hermes", "skills", "uninstall", "test-skill"],
    )

    assert captured["name"] == "test-skill"
    assert captured["skip_confirm"] is False