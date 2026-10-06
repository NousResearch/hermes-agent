"""`hermes curator ledger --show <id>` prints one entry with a blob-reconstructed diff."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def ledger_env(tmp_path, monkeypatch):
    from tools import skill_ledger

    home = tmp_path / "home"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(skill_ledger, "get_hermes_home", lambda: home)
    return home


def _append_patch_entry(env) -> str:
    """One real ledger row: patch with before/after blobs under the isolated home."""
    from tools import skill_ledger

    path = str(env / "skills" / "my-skill" / "SKILL.md")
    before = [{"path": path, "sha256": skill_ledger._store_blob(b"original body\n")}]
    after = [{"path": path, "sha256": skill_ledger._store_blob(b"patched body\n")}]
    entry_id = skill_ledger.append_entry(
        "patch", "my-skill", before=before, after=after,
        evidence={"session_id": "20261006_112750_3e2e14c8", "file_path": "SKILL.md"})
    assert entry_id is not None
    return entry_id


def test_show_prints_header_evidence_and_diff(ledger_env, capsys):
    import hermes_cli.curator as curator_cli

    entry_id = _append_patch_entry(ledger_env)
    rc = curator_cli._cmd_ledger(_ns(compact=False, show=entry_id, skill=None, limit=20))
    assert rc == 0
    out = capsys.readouterr().out
    assert f"entry {entry_id}" in out
    assert "actor=" in out and "action=patch" in out and "skill=my-skill" in out
    assert "session_id: 20261006_112750_3e2e14c8" in out
    assert "-original body" in out and "+patched body" in out


def test_show_without_content_changes_says_so(ledger_env, capsys):
    import hermes_cli.curator as curator_cli
    from tools import skill_ledger

    entry_id = skill_ledger.append_entry("patch", "my-skill", before=[], after=[])
    rc = curator_cli._cmd_ledger(_ns(compact=False, show=entry_id, skill=None, limit=20))
    assert rc == 0
    assert "no recoverable content changes" in capsys.readouterr().out


def test_show_unknown_id_fails_with_stderr(ledger_env, capsys):
    import hermes_cli.curator as curator_cli

    rc = curator_cli._cmd_ledger(_ns(compact=False, show="deadbeefdead", skill=None, limit=20))
    assert rc == 1
    assert "no ledger entry" in capsys.readouterr().err


def test_ledger_argparse_registers_show(ledger_env):
    """`--show` is wired into the `ledger` subcommand parser."""
    import argparse

    import hermes_cli.curator as curator_cli

    parser = argparse.ArgumentParser(prog="hermes curator")
    curator_cli.register_cli(parser)
    args = parser.parse_args(["ledger", "--show", "abc123"])
    assert args.show == "abc123" and args.compact is False


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)
