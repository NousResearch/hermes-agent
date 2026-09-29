"""Connection presence: read from files and credentials, reported as presence only."""

from agent import connection_status as cs


def test_reports_presence_per_connection_and_never_a_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cs.Path, "home", lambda: tmp_path)
    (tmp_path / "google_token.json").write_text("{}", encoding="utf-8")
    env = {"NOTION_API_KEY": "ntn_secret", "TELEGRAM_BOT_TOKEN": "  "}

    got = cs.snapshot(env.get, {"mcp_servers": {"canva": {}}})

    assert got["google"] == cs.CONNECTED and got["notion"] == cs.CONNECTED and got["mcp"] == cs.CONNECTED
    # A blank credential is not a connection; nothing here is a value.
    assert got["messaging"] == cs.MISSING and got["github"] == cs.MISSING and got["email"] == cs.MISSING
    assert "ntn_secret" not in str(got)


def test_what_local_state_cannot_tell_is_unknown_not_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cs.Path, "home", lambda: tmp_path)

    assert cs.snapshot(lambda k: None, {})["phone"] == cs.UNKNOWN
