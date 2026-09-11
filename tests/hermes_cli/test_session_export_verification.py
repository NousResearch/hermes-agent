"""Export metadata must not consume or rewrite conversation content."""

import sys
from pathlib import Path

import pytest

from hermes_cli.session_export_md import (
    render_session_markdown,
    verify_export_file,
    write_session_markdown,
)
from hermes_state import SessionDB


def _previous_export(fmt):
    return render_session_markdown({
        "id": "previous-session",
        "title": "Previous conversation",
        "message_count": 2,
        "messages": [
            {"role": "user", "content": "An earlier conversation."},
            {"role": "assistant", "content": "An earlier reply."},
        ],
    }, fmt=fmt)


@pytest.mark.parametrize("fmt", ["md", "qmd"])
@pytest.mark.parametrize("message_kind", ["plain", "previous_export", "placeholder"])
def test_export_preserves_message_markers_and_detects_tampering(tmp_path, fmt, message_kind):
    contents = {
        "plain": "A normal conversation.",
        "previous_export": "Explain this previous export:\n\n" + _previous_export(fmt),
        "placeholder": "Explain `__SHA256_PLACEHOLDER__` and the unquoted __SHA256_PLACEHOLDER__ token.",
    }
    content = contents[message_kind] + "\nKeep this trailing message text."
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("current-session", source="cli")
        db.append_message("current-session", role="user", content=content)
        session = db.export_session("current-session")
        path = write_session_markdown(session, tmp_path / "exports", fmt=fmt)
        exported = path.read_text(encoding="utf-8")

        assert content in exported
        assert verify_export_file(path, session) == (True, "ok")
        assert verify_export_file(path, {**session, "id": "previous-session"}) == (False, "session id mismatch")
        wrong_count = {**session, "messages": [*session["messages"], {"role": "assistant", "content": "Extra."}]}
        assert verify_export_file(path, wrong_count) == (False, "message count mismatch")
        assert content in render_session_markdown(session, fmt=fmt, include_verification=False)
        assert db.get_messages("current-session")[0]["content"] == content

        path.write_text(exported.replace("Keep this trailing message text.", "Altered message."), encoding="utf-8")
        assert verify_export_file(path, session) == (False, "sha256 mismatch")


@pytest.mark.parametrize("fmt", ["md", "qmd"])
def test_cli_verified_export_accepts_nested_export_without_losing_content(tmp_path, monkeypatch, capsys, fmt):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    content = _previous_export(fmt) + "\nExplain the `__SHA256_PLACEHOLDER__` token."
    with SessionDB() as db:
        db.create_session("export-target", source="cli")
        db.append_message("export-target", role="user", content=content)
        db.create_session("keep-session", source="cli")
        db.append_message("keep-session", role="user", content="Keep this other conversation.")
        exported_session = db.export_session("export-target")

    output_dir = tmp_path / "exports"
    monkeypatch.setattr(sys, "argv", [
        "hermes", "sessions", "export", "--format", fmt,
        "--session-id", "export-target", "--delete-after-verified", "--yes", str(output_dir),
    ])
    main_mod.main()

    output = capsys.readouterr().out
    assert "Export verification failed" not in output
    path, = output_dir.glob(f"*.{fmt}")
    assert content in path.read_text(encoding="utf-8")
    assert verify_export_file(path, exported_session) == (True, "ok")
    with SessionDB() as db:
        assert db.get_session("export-target") is None
        assert db.get_messages("keep-session")[0]["content"] == "Keep this other conversation."
