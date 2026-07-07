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


def test_sessions_export_md_reports_unknown_session(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    output_dir = tmp_path / "exports"

    class FakeDB:
        def resolve_session_id(self, session_id):
            return None

        def export_session(self, session_id):
            raise AssertionError("export_session should not be called")

        def close(self):
            pass

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
            "missing",
            str(output_dir),
        ],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert "Session 'missing' not found." in output
    assert not output_dir.exists()


def test_sessions_export_md_supports_qmd_format(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def resolve_session_id(self, session_id):
            return "s1"

        def export_session(self, session_id):
            return {"id": "s1", "title": "QMD", "messages": []}

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "sessions",
            "export",
            "--session-id",
            "s1",
            "--format",
            "qmd",
            str(tmp_path),
        ],
    )

    main_mod.main()

    assert len(list(tmp_path.glob("*.qmd"))) == 1
    assert "Exported 1 session" in capsys.readouterr().out


def test_sessions_export_md_rejects_stdout_target(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def resolve_session_id(self, session_id):
            raise AssertionError("md export to stdout must be refused before DB access")

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "export", "--format", "md", "--session-id", "s1", "-"],
    )

    main_mod.main()

    assert "only supported with --format jsonl" in capsys.readouterr().out


def test_sessions_export_jsonl_requires_output_path(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def export_all(self, **kwargs):
            raise AssertionError("jsonl export without an output path must be refused")

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "export"])

    main_mod.main()

    assert "requires an output path" in capsys.readouterr().out


def test_sessions_export_md_bulk_dry_run_lists_candidates(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def list_prune_candidates(self, **kwargs):
            # Export flows through the shared prune-filter machinery:
            # --older-than 30 becomes a started_before epoch bound, source
            # passes through, and archived is tri-state None (export includes
            # archived sessions).
            assert kwargs.get("source") == "cron"
            assert kwargs.get("started_before") is not None
            assert kwargs.get("archived") is None
            return [{"id": "s1", "source": "cron"}, {"id": "s2", "source": "cron"}]

        def close(self):
            pass

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
            "--older-than",
            "30",
            "--source",
            "cron",
            "--dry-run",
            str(tmp_path),
        ],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert "Would export 2 session(s)" in output
    assert "s1" in output
    assert "s2" in output
    assert not list(tmp_path.glob("*.md"))


def test_sessions_export_md_bulk_requires_filter(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def list_prune_candidates(self, **kwargs):
            raise AssertionError("bulk export without filters should refuse")

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "export", "--format", "md", str(tmp_path)],
    )

    main_mod.main()

    assert "Refusing bulk export without a filter" in capsys.readouterr().out


def test_sessions_export_md_bulk_writes_manifest(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def list_prune_candidates(self, **kwargs):
            return [{"id": "s1"}, {"id": "s2"}]

        def export_session_lineage(self, session_id):
            return {"id": session_id, "title": session_id, "messages": [{"role": "user", "content": session_id}]}

        def close(self):
            pass

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
            "--older-than",
            "90",
            "--lineage",
            "logical",
            str(tmp_path),
        ],
    )

    main_mod.main()

    assert len(list(tmp_path.glob("*.md"))) == 2
    manifest = tmp_path / "manifest.jsonl"
    assert manifest.exists()
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "Exported 2 session(s)" in capsys.readouterr().out


def test_sessions_export_md_delete_after_verified_requires_yes(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def close(self):
            pass

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
            "s1",
            "--delete-after-verified",
            str(tmp_path),
        ],
    )

    main_mod.main()

    assert "requires --yes" in capsys.readouterr().out


def test_sessions_export_md_delete_after_verified_deletes_after_file_check(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            return "s1"

        def export_session(self, session_id):
            return {"id": "s1", "title": "Delete", "message_count": 1, "messages": [{"role": "user", "content": "safe"}]}

        def delete_session(self, session_id, **kwargs):
            captured["deleted"] = session_id
            return True

        def close(self):
            pass

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
            "s1",
            "--delete-after-verified",
            "--yes",
            str(tmp_path),
        ],
    )

    main_mod.main()

    assert captured == {"deleted": "s1"}
    assert len(list(tmp_path.glob("*.md"))) == 1
    assert "Deleted exported session 's1'" in capsys.readouterr().out


def test_sessions_export_md_accepts_duration_age_grammar(monkeypatch, tmp_path, capsys):
    """--older-than accepts the same AGE grammar as prune ('2w', '5h', ISO)."""
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def list_prune_candidates(self, **kwargs):
            assert kwargs.get("started_before") is not None
            return [{"id": "s1", "source": "cli"}]

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes", "sessions", "export", "--format", "md",
            "--older-than", "2w", "--dry-run", str(tmp_path),
        ],
    )

    main_mod.main()

    assert "Would export 1 session(s)" in capsys.readouterr().out


def test_sessions_export_md_supports_extended_prune_filters(monkeypatch, tmp_path, capsys):
    """Filters like --model/--min-messages pass through the shared machinery."""
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def list_prune_candidates(self, **kwargs):
            captured.update(kwargs)
            return []

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes", "sessions", "export", "--format", "md",
            "--model", "sonnet", "--min-messages", "5", "--dry-run",
            str(tmp_path),
        ],
    )

    main_mod.main()

    assert captured.get("model_like") == "sonnet"
    assert captured.get("min_messages") == 5
    assert "Would export 0 session(s)" in capsys.readouterr().out


def test_sessions_export_jsonl_honors_filters(monkeypatch, tmp_path, capsys):
    """JSONL bulk export uses the same filter machinery as md/qmd."""
    import json

    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def list_prune_candidates(self, **kwargs):
            assert kwargs.get("source") == "telegram"
            return [{"id": "s1", "source": "telegram"}]

        def export_session(self, session_id):
            return {"id": session_id, "messages": [{"role": "user", "content": "hi"}]}

        def export_all(self, **kwargs):
            raise AssertionError("filtered jsonl export must not fall back to export_all")

        def close(self):
            pass

    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "export", "--source", "telegram", str(out)],
    )

    main_mod.main()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "s1"
    assert "Exported 1 sessions" in capsys.readouterr().out


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
