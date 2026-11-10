"""Configurable session-import limits: sessions.import_max_* in config.yaml."""

import pytest

from hermes_state import SessionDB


def test_import_limit_defaults_are_the_import_guard_constants():
    """Registered config defaults and the SessionDB fallback constants are one contract."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    sessions = DEFAULT_CONFIG["sessions"]
    assert sessions["import_max_sessions"] == SessionDB._IMPORT_MAX_SESSIONS
    assert sessions["import_max_messages_per_session"] == SessionDB._IMPORT_MAX_MESSAGES_PER_SESSION
    assert sessions["import_max_total_messages"] == SessionDB._IMPORT_MAX_TOTAL_MESSAGES
    assert sessions["import_max_session_bytes"] == SessionDB._IMPORT_MAX_SESSION_BYTES
    assert sessions["import_max_total_bytes"] == SessionDB._IMPORT_MAX_TOTAL_BYTES


@pytest.mark.parametrize("raw", ["-1", "abc"])
def test_invalid_import_limit_falls_back_to_default(tmp_path, monkeypatch, raw):
    """A negative or non-numeric sessions.import_max_* value keeps the default guard rather
    than disabling it (0 is the only off switch) or rejecting every payload."""
    from hermes_state import resolved_import_max_messages_per_session

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"sessions:\n  import_max_messages_per_session: {raw}\n")
    assert resolved_import_max_messages_per_session() == SessionDB._IMPORT_MAX_MESSAGES_PER_SESSION


@pytest.mark.parametrize(
    ("key", "payload", "error"),
    [
        ("import_max_sessions", "sessions", "sessions must contain at most"),
        ("import_max_messages_per_session", "messages", "messages exceeds the per-session import limit"),
        ("import_max_total_messages", "total_messages", "messages exceeds the total import limit"),
        ("import_max_session_bytes", "bytes", "session exceeds the import size limit"),
        ("import_max_total_bytes", "total_bytes", "import exceeds the total size limit"),
    ],
)
def test_import_limits_read_from_config_yaml(tmp_path, monkeypatch, key, payload, error):
    """Each sessions.import_max_* key, set in a real config.yaml, governs its guard: a small
    value rejects a payload the default accepts, and 0 disables the guard."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def sessions_payload(prefix):
        ids = [f"{prefix}-{i}" for i in range(2)]
        if payload == "sessions":
            return [{"id": sid, "messages": []} for sid in ids]
        if payload == "messages":
            return [{"id": ids[0], "messages": [{"role": "user", "content": "x"}] * 3}]
        if payload == "total_messages":
            return [{"id": sid, "messages": [{"role": "user", "content": "x"}] * 2} for sid in ids]
        content = "x" * 400
        if payload == "bytes":
            return [{"id": ids[0], "messages": [{"role": "user", "content": content}]}]
        return [{"id": sid, "messages": [{"role": "user", "content": content}]} for sid in ids]

    def run_import(limit, prefix):
        config = f"sessions:\n  {key}: {limit}\n" if limit is not None else "sessions: {}\n"
        (home / "config.yaml").write_text(config)
        db = SessionDB(db_path=tmp_path / f"{prefix}.db")
        try:
            return db.import_sessions(sessions_payload(prefix))
        except ValueError as exc:
            return {"ok": False, "errors": [{"error": str(exc)}]}
        finally:
            db.close()

    narrow = 2 if payload in {"messages", "total_messages"} else 1 if payload == "sessions" else 300
    narrowed = run_import(narrow, "narrow")
    assert narrowed["ok"] is False
    assert narrowed["errors"][0]["error"].startswith(error)

    # With both default sources narrowed too (the registered config default and the SessionDB
    # fallback), an unset key rejects the payload and 0 still accepts it, so the disabled case
    # proves the guard is off rather than merely not reached.
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    monkeypatch.setitem(DEFAULT_CONFIG["sessions"], key, narrow)
    monkeypatch.setattr(SessionDB, "_IMPORT_MAX_" + key.removeprefix("import_max_").upper(), narrow)
    defaulted = run_import(None, "defaulted")
    assert defaulted["ok"] is False
    assert defaulted["errors"][0]["error"].startswith(error)
    assert run_import(0, "disabled")["ok"] is True
