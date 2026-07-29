"""
Tests for --personal / --user flag on `hermes skills list`.

--personal / --user → args.personal → do_list(personal_only=True): show local
skills explicitly marked user-specific or recorded as agent-created, instead
of every locally installed skill.
"""

import sys


def _run_skills_list(monkeypatch, argv):
    from hermes_cli.main import main

    captured = {}

    def fake_skills_command(args):
        captured["personal"] = getattr(args, "personal", None)

    monkeypatch.setattr("hermes_cli.skills_hub.skills_command", fake_skills_command)
    monkeypatch.setattr(sys, "argv", argv)

    main()
    return captured


def test_cli_skills_list_personal_flag(monkeypatch):
    captured = _run_skills_list(monkeypatch, ["hermes", "skills", "list", "--personal"])

    assert captured["personal"] is True


def test_cli_skills_list_user_alias(monkeypatch):
    """--user must behave identically to --personal."""
    captured = _run_skills_list(monkeypatch, ["hermes", "skills", "list", "--user"])

    assert captured["personal"] is True


def test_cli_skills_list_personal_defaults_false(monkeypatch):
    captured = _run_skills_list(monkeypatch, ["hermes", "skills", "list"])

    assert captured["personal"] is False
