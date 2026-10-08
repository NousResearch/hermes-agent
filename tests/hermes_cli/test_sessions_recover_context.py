"""Reviewed offline summaries recover blocked context without deleting history (#90177)."""

import argparse

import hermes_state

from agent.secret_scope import reset_multiplex_context, set_multiplex_context
from hermes_cli.sessions_cmd import cmd_sessions
from hermes_cli.subcommands.sessions import build_sessions_parser
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB


def _args(session_id, summary_file, *, apply=False):
    parser = argparse.ArgumentParser()
    build_sessions_parser(parser.add_subparsers(dest="command"), cmd_sessions=cmd_sessions)
    argv = ["sessions", "recover-context", session_id, "--summary-file", str(summary_file)]
    return parser.parse_args(argv + (["--apply"] if apply else []))


def test_reviewed_summary_preserves_history_prompt_and_profile_scope(tmp_path, capsys, monkeypatch):
    # Restore the native call-time resolver; the hermetic fixture pins its own DB path.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
    original = "Log excerpt: [Previous line repeated 8 more times]"
    ids = {}
    for home in homes:
        home.mkdir()
        with SessionDB(db_path=home / "state.db") as db:
            db.create_session("root", "cli", system_prompt=f"Pinned prompt for {home.name}")
            db.append_message("root", "user", original)
            db.append_message("root", "assistant", f"Work in {home.name}")
            db.end_session("root", "compression")
            db.create_session("tip", "cli", parent_session_id="root",
                              system_prompt=f"Pinned prompt for {home.name}")
            db.append_message("tip", "user", original)
            db.append_message("tip", "assistant", "Pending task")
            ids[home] = db.get_active_message_ids("tip")

    for index, home in enumerate([homes[0], homes[1], homes[0]]):
        summary_file = home / "summary.txt"
        summary = f"Continue the pending task in {home.name}; recovery {index}."
        summary_file.write_bytes(
            (b"\xef\xbb\xbf" if index == 1 else b"") + summary.encode("utf-8"),
        )
        token = set_hermes_home_override(home)
        multiplex_token = set_multiplex_context(True)
        try:
            with SessionDB() as db:
                before = db.get_messages("tip")
            assert cmd_sessions(_args("roo", summary_file)) == 0
            assert "Preview" in capsys.readouterr().out
            with SessionDB() as db:
                assert db.get_messages("tip") == before
            assert cmd_sessions(_args("root", summary_file, apply=True)) == 0
            capsys.readouterr()
            with SessionDB() as db:
                model, display = db.get_resume_conversations("tip")
                assert [row["role"] for row in model] == ["user", "assistant"]
                assert summary in model[0]["content"]
                assert "\ufeff" not in model[0]["content"]
                assert model[0]["_compressed_summary"]
                assert not any(original in str(row["content"]) for row in model)
                assert any(original in str(row["content"]) for row in display)
                assert db.get_session("tip")["system_prompt"] == f"Pinned prompt for {home.name}"
                assert db.get_session("tip")["message_count"] == len(model)
                assert db.get_session("tip")["tool_call_count"] == 0
                assert any(row["session_id"] == "tip" for row in db.search_messages("Previous"))
                archived = db._read_all(
                    "SELECT active, compacted FROM messages WHERE id IN (?, ?)", ids[home],
                )
                assert all(not row["active"] and row["compacted"] for row in archived)
                db.append_message("tip", "user", "Continue")
                model, _ = db.get_resume_conversations("tip")
                assert [row["role"] for row in model] == ["user", "assistant", "user"]

            with SessionDB() as db:
                current = db.get_messages("tip")
            summary_file.write_text(" \n", encoding="utf-8")
            assert cmd_sessions(_args("root", summary_file, apply=True)) == 2
            assert cmd_sessions(_args("root", home / "missing.txt", apply=True)) == 2
            summary_file.write_bytes(b"\xff")
            assert cmd_sessions(_args("root", summary_file, apply=True)) == 2
            summary_file.write_text(summary, encoding="utf-8")
            assert cmd_sessions(_args("missing", summary_file, apply=True)) == 1
            if index < 2:
                with SessionDB() as db:
                    db.create_session("empty", "cli")
                    db.create_session("tip-other", "cli")
            assert cmd_sessions(_args("empty", summary_file, apply=True)) == 1
            assert cmd_sessions(_args("ti", summary_file, apply=True)) == 1
            assert cmd_sessions(_args("", summary_file, apply=True)) == 2
            capsys.readouterr()
            with SessionDB() as db:
                assert db.get_messages("tip") == current
        finally:
            reset_multiplex_context(multiplex_token)
            reset_hermes_home_override(token)

    with SessionDB(db_path=homes[1] / "state.db") as db:
        model, _ = db.get_resume_conversations("tip")
        assert "recovery 1" in model[0]["content"]
        assert "recovery 2" not in model[0]["content"]
