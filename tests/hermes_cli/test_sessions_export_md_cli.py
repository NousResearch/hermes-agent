import sys


def test_sessions_export_md_writes_single_session(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            captured["resolved_from"] = session_id
            return "20260706_123456_abcd1234"

        def export_session(self, session_id):
            captured["exported"] = session_id
            return {
                "id": session_id,
                "title": "Export CLI Test",
                "source": "cli",
                "message_count": 1,
                "messages": [{"role": "user", "content": "hello"}],
            }

        def delete_session(self, *args, **kwargs):
            raise AssertionError("markdown export must not delete sessions")

        def prune_sessions(self, *args, **kwargs):
            raise AssertionError("markdown export must not prune sessions")

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "sessions",
            "export",
            "--format",
            "md",
            "--session-id",
            "20260706_123456",
            str(tmp_path),
        ],
    )

    main_mod.main()

    output = capsys.readouterr().out
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "# Export CLI Test" in text
    assert "hello" in text
    assert captured == {
        "resolved_from": "20260706_123456",
        "exported": "20260706_123456_abcd1234",
        "closed": True,
    }
    assert "Exported 1 session" in output
    assert "1 message" in output
    assert str(files[0]) in output


def test_sessions_export_redact_scrubs_secrets(monkeypatch, tmp_path):
    """--redact runs exported content through force-mode secret redaction."""
    import hermes_cli.main as main_mod
    import hermes_state

    secret = "sk-proj-Zz12345678901234567890123456789012345678"

    class FakeDB:
        def resolve_session_id(self, session_id):
            return "s1"

        def export_session(self, session_id):
            return {
                "id": "s1",
                "title": "Redact",
                "messages": [
                    {"role": "tool", "name": "terminal", "content": f"api key: {secret}"}
                ],
            }

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes", "sessions", "export", "--format", "md",
            "--session-id", "s1", "--redact", str(tmp_path),
        ],
    )

    main_mod.main()

    text = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert secret not in text
    assert "api key:" in text


def _trace_fake_db(captured):
    class FakeDB:
        def resolve_session_id(self, session_id):
            return "s1"

        def get_session(self, session_id):
            return {"id": session_id, "model": "test-model"}

        def get_messages_as_conversation(self, session_id):
            captured["conv"] = session_id
            return [
                {"role": "user", "content": "hello trace"},
                {"role": "assistant", "content": "hi"},
            ]

        def close(self):
            captured["closed"] = True

    return FakeDB()


def test_sessions_export_trace_upload_routes_to_uploader(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state
    from agent import trace_upload as trace_mod

    captured = {}
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: _trace_fake_db(captured))

    def fake_upload(session_id, **kwargs):
        captured["uploaded"] = session_id
        captured["kwargs"] = kwargs
        return "Uploaded -> https://example.test/dataset"

    monkeypatch.setattr(trace_mod, "upload_session_trace", fake_upload)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes", "sessions", "export", "--format", "trace",
            "--session-id", "s1", "--upload", "--public",
        ],
    )

    main_mod.main()

    assert captured["uploaded"] == "s1"
    assert captured["kwargs"]["private"] is False
    assert captured["kwargs"]["redact"] is True
    assert "Uploaded ->" in capsys.readouterr().out


def test_sessions_export_trace_only_flag_rejected(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: _trace_fake_db(captured))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes", "sessions", "export", "--format", "trace",
            "--session-id", "s1", "--only", "user-prompts", "-",
        ],
    )

    main_mod.main()

    assert "--only user-prompts supports --format jsonl or md." in capsys.readouterr().out
