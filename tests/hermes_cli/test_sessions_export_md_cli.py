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

    monkeypatch.setattr(hermes_state, "SessionDB", lambda *args, **kwargs: FakeDB())
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

    monkeypatch.setattr(hermes_state, "SessionDB", lambda *args, **kwargs: FakeDB())
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
