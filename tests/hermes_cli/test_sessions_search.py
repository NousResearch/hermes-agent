import json
import sys


class _DBFactory:
    def __init__(self, db):
        self.db = db

    def __call__(self):
        return self.db


def _seed_search_db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s_alpha", source="cli")
    db.set_session_title("s_alpha", "Alpha Rollout")
    db.append_message("s_alpha", role="user", content="Plan the nebula rollout checklist")
    db.append_message("s_alpha", role="assistant", content="Nebula rollout has three gates.")

    db.create_session("s_beta", source="telegram")
    db.set_session_title("s_beta", "Beta Notes")
    db.append_message("s_beta", role="user", content="Discuss unrelated billing notes")
    return db


def test_sessions_search_prints_transcript_matches(monkeypatch, capsys, tmp_path):
    import hermes_cli.main as main_mod
    import hermes_state

    db = _seed_search_db(tmp_path)
    monkeypatch.setattr(hermes_state, "SessionDB", _DBFactory(db))
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "search", "nebula", "--limit", "3"],
    )

    main_mod.main()

    out = capsys.readouterr().out
    assert 'Found 1 matching session(s) for "nebula"' in out
    assert "Alpha Rollout" in out
    assert "s_alpha" in out
    assert "snippet:" in out
    assert "hermes --resume s_alpha" in out


def test_sessions_search_json_outputs_tool_payload(monkeypatch, capsys, tmp_path):
    import hermes_cli.main as main_mod
    import hermes_state

    db = _seed_search_db(tmp_path)
    monkeypatch.setattr(hermes_state, "SessionDB", _DBFactory(db))
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "search", "nebula", "--json"],
    )

    main_mod.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["mode"] == "discover"
    assert payload["query"] == "nebula"
    assert payload["results"][0]["session_id"] == "s_alpha"


def test_sessions_search_empty_result_has_human_message(monkeypatch, capsys, tmp_path):
    import hermes_cli.main as main_mod
    import hermes_state

    db = _seed_search_db(tmp_path)
    monkeypatch.setattr(hermes_state, "SessionDB", _DBFactory(db))
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "search", "missingterm"],
    )

    main_mod.main()

    out = capsys.readouterr().out
    assert 'No sessions matched "missingterm"' in out
    assert "FTS5 syntax" in out
