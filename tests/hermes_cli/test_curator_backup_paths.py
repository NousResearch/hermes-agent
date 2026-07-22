"""`hermes curator backup`/`rollback` paths must be profile-aware.

Regression: the CLI hardcoded ``~/.hermes/skills/...`` in the snapshot-created
message, the rollback confirmation message, and the backup/rollback help text.
Under a profile (or on native Windows) the real tree is
``$HERMES_HOME/skills/...`` — e.g. ``~/.hermes/profiles/coder/skills/...`` — so
the printed path was wrong. The fix routes every user-facing path through
``display_hermes_home()`` (AGENTS.md "Profiles" rule #2).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _use_profile(monkeypatch, tmp_path) -> str:
    """Activate a ``coder`` profile under a fake home and return the expected
    ``display_hermes_home()`` string (computed the same way the code does, so
    the assertions hold on Windows path separators too)."""
    from hermes_constants import display_hermes_home

    home = tmp_path
    hermes_home = home / ".hermes" / "profiles" / "coder"
    hermes_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return display_hermes_home()


def _subcommand_help(parser: argparse.ArgumentParser, name: str):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for pseudo in action._choices_actions:
                if pseudo.dest == name:
                    return pseudo.help
    return None


def test_backup_message_is_profile_aware(monkeypatch, capsys, tmp_path):
    import hermes_cli.curator as curator_cli
    import agent.curator_backup as curator_backup

    dhh = _use_profile(monkeypatch, tmp_path)
    monkeypatch.setattr(curator_backup, "is_enabled", lambda: True)
    monkeypatch.setattr(
        curator_backup, "snapshot_skills",
        lambda reason="manual": SimpleNamespace(name="2026-06-06T00-00-00Z"),
    )

    rc = curator_cli._cmd_backup(_ns(reason=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert f"{dhh}/skills/.curator_backups/2026-06-06T00-00-00Z" in out
    # Must not regress to the hardcoded default-profile path.
    assert "snapshot created at ~/.hermes/skills/.curator_backups" not in out


def test_rollback_confirmation_is_profile_aware(monkeypatch, capsys, tmp_path):
    import hermes_cli.curator as curator_cli
    import agent.curator_backup as curator_backup

    dhh = _use_profile(monkeypatch, tmp_path)
    target = tmp_path / "snap" / "2026-06-06T00-00-00Z"
    monkeypatch.setattr(curator_backup, "_resolve_backup", lambda backup_id: target)
    monkeypatch.setattr(curator_backup, "_read_manifest", lambda path: {})
    # Decline at the prompt so rollback() is never reached — we only care
    # about the confirmation message's path.
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    rc = curator_cli._cmd_rollback(_ns(list=False, backup_id=None, yes=False))
    assert rc == 1
    out = capsys.readouterr().out
    assert f"replace the current {dhh}/skills/ tree" in out
    assert "replace the current ~/.hermes/skills/ tree" not in out


def test_backup_rollback_help_is_profile_aware(monkeypatch, tmp_path):
    import hermes_cli.curator as curator_cli

    dhh = _use_profile(monkeypatch, tmp_path)
    parser = argparse.ArgumentParser(prog="hermes curator")
    curator_cli.register_cli(parser)

    backup_help = _subcommand_help(parser, "backup")
    rollback_help = _subcommand_help(parser, "rollback")
    assert backup_help is not None and rollback_help is not None
    assert f"{dhh}/skills/" in backup_help
    assert f"{dhh}/skills/" in rollback_help
    assert "~/.hermes/skills/" not in backup_help
    assert "~/.hermes/skills/" not in rollback_help
