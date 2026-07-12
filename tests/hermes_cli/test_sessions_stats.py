import sys


def test_sessions_stats_includes_dynamically_discovered_source(
    monkeypatch, capsys, _isolate_hermes_home
):
    import hermes_cli.main as main_mod
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session("matrix-session", source="matrix", model="test/model")
    finally:
        db.close()

    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "stats"])

    main_mod.main()

    assert "  matrix: 1 sessions" in capsys.readouterr().out.splitlines()
