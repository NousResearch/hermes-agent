import sys

from persistence_assembly import open_persistence, open_persistence_reader, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


def test_sessions_export_md_writes_single_session(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    sid = "20260706_123456_abcd1234"
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, sid, "cli", title="Export CLI Test")
        seed_messages(db, sid, {"role": "user", "content": "hello"})
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", [
        "hermes", "sessions", "export", "--format", "md",
        "--session-id", "20260706_123456", str(tmp_path / "exports"),
    ])

    main_mod.main()

    output = capsys.readouterr().out
    files = list((tmp_path / "exports").glob("*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "# Export CLI Test" in text
    assert "hello" in text
    assert path.read_bytes() == before
    assert "Exported 1 session" in output
    assert "1 message" in output
    assert str(files[0]) in output


def test_sessions_export_verified_delete_keeps_file(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    from hermes_cli.session_export import to_legacy_export_dict
    from hermes_cli.session_export_md import verify_export_file

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, "delete-me", "cli")
        seed_messages(db, "delete-me", {"role": "user", "content": "retain this exported payload"})
        export = db.maintenance.export_session("delete-me")
        assert export is not None
        original = to_legacy_export_dict(export)
    monkeypatch.setattr(sys, "argv", [
        "hermes", "sessions", "export", "--format", "md",
        "--session-id", "delete-me", "--delete-after-verified", "--yes",
        str(tmp_path / "exports"),
    ])

    main_mod.main()

    files = list((tmp_path / "exports").glob("*.md"))
    assert len(files) == 1
    assert verify_export_file(files[0], original)[0]
    with open_persistence_reader(path) as db:
        assert db.sessions.get("delete-me") is None
    assert "Deleted exported session" in capsys.readouterr().out


def test_sessions_export_redact_scrubs_secrets(monkeypatch, tmp_path):
    """Redaction affects the exported file, never the stored transcript."""
    import hermes_cli.main as main_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    secret = "sk-proj-Zz12345678901234567890123456789012345678"
    path = tmp_path / "state.db"
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, "s1", "cli")
        seed_messages(db, "s1", {"role": "tool", "content": f"api key: {secret}", "tool_name": "terminal"})
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", [
        "hermes", "sessions", "export", "--format", "md",
        "--session-id", "s1", "--redact", str(tmp_path / "exports"),
    ])

    main_mod.main()

    files = list((tmp_path / "exports").glob("*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert secret not in text
    assert "api key:" in text
    assert path.read_bytes() == before
