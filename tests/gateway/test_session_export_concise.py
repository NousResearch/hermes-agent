"""``session.export`` with ``concise``, and the CLI's ``/save --concise`` / ``/copy all`` on top of it.

``export_session`` is called directly with a real ``SessionDB`` behind a stub authority (the websocket
owner harness is Linux-only), so these run on every host.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import SUMMARY_PREFIX
from cli import HermesCLI
from gateway import session_tui_ops
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError


class _AnySession(dict):
    def __missing__(self, key):
        return SimpleNamespace(source=None)


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


@pytest.fixture
def rpc(db, tmp_path, monkeypatch):
    """``session.export`` as the gateway runs it, over *db*; returns ``call(**params)``."""
    # ``_live_ref`` indexes the owner's live sessions by id: any id resolves to a session with no platform source.
    authority = SimpleNamespace(db=db, profile_id=str(tmp_path), authorize=lambda *a, **k: None,
                                sessions=_AnySession())
    connection = SimpleNamespace(authority=authority, actor=None)
    monkeypatch.setattr("gateway.session_ancillary.physical_target", lambda authority, ref: ref.session_id)

    def call(**params):
        ref = SimpleNamespace(session_id=params.get("session_id"))
        return asyncio.run(session_tui_ops.export_session(connection, ref, params))

    return call


def _chat(db, sid="chat"):
    db.create_session(sid, source="desktop")
    for i in range(3):
        db.append_message(sid, "user", f"question {i}")
        db.append_message(sid, "assistant", f"answer {i}", reasoning="HIDDEN-THOUGHT")
        db.append_message(sid, "tool", "TOOL-OUTPUT", tool_name="terminal", tool_call_id=f"c{i}")
    live = db.get_messages_as_conversation(sid)
    db.archive_and_compact(
        sid, [{"role": "user", "content": f"{SUMMARY_PREFIX} summary"}] + live[-2:], tail_count=2)
    db.append_message(sid, "user", "after compaction")
    db.append_message(sid, "assistant", "last answer")
    return sid


def test_concise_export_is_the_whole_visible_dialogue_not_the_tool_log(db, rpc):
    sid = _chat(db)

    out = rpc(session_id=sid, format="md", concise=True)

    assert out["format"] == "md" and out["filename"].endswith(".md")
    md = out["content"]
    assert md.startswith("# Conversation: ")
    for i in range(3):
        assert f"question {i}" in md and f"answer {i}" in md  # pre-compaction turns are still there
    assert "after compaction" in md and "last answer" in md
    assert "TOOL-OUTPUT" not in md and "HIDDEN-THOUGHT" not in md and "summary" not in md


def test_default_exports_keep_their_meaning(db, rpc):
    sid = _chat(db)

    plain_md = rpc(session_id=sid, format="md")["content"]
    as_json = rpc(session_id=sid)

    assert "TOOL-OUTPUT" in plain_md  # the normal markdown export still carries tool records
    assert as_json["format"] == "json" and "TOOL-OUTPUT" in as_json["content"]
    assert rpc(session_id=sid, format="md", concise=False)["content"] == plain_md


@pytest.mark.parametrize("params", [
    {"format": "json", "concise": True},   # concise is Markdown only
    {"format": "html", "concise": True},
    {"format": "md", "concise": "yes"},    # not a boolean
])
def test_concise_misuse_is_refused_not_guessed(db, rpc, params):
    sid = _chat(db)

    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        rpc(session_id=sid, **params)


def test_unknown_session_and_redaction(db, rpc):
    with pytest.raises(RuntimeStoreError, match="not_found"):
        rpc(session_id="missing", format="md", concise=True)
    db.create_session("sec", source="desktop")
    db.append_message("sec", "user", "my key is sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD")
    db.append_message("sec", "assistant", "noted")

    assert "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD" in rpc(session_id="sec", format="md", concise=True)["content"]
    assert "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD" not in rpc(
        session_id="sec", format="md", concise=True, redact=True)["content"]


def test_export_over_the_size_cap_is_refused_never_truncated(db, rpc, monkeypatch):
    sid = _chat(db)
    monkeypatch.setattr("hermes_state.resolved_max_export_messages", lambda: 2)

    with pytest.raises(RuntimeStoreError, match="export_too_large"):
        rpc(session_id=sid, format="md", concise=True)


def test_export_is_read_only(db, rpc):
    sid = _chat(db)
    before = (db.get_messages(sid, include_inactive=True), db.get_session(sid))

    rpc(session_id=sid, format="md", concise=True)

    assert (db.get_messages(sid, include_inactive=True), db.get_session(sid)) == before


# --- the classic CLI on top -------------------------------------------------------------------------


def _cli(db, sid):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = sid
    cli._session_db = db
    cli.conversation_history = [{"role": "user", "content": "only the live tail is in memory"}]
    cli._app = None
    return cli


def test_save_concise_writes_the_same_text_as_the_rpc(db, rpc, tmp_path, monkeypatch, capsys):
    sid = _chat(db)
    monkeypatch.setattr("hermes_cli.gateway_tools_notify.gateway_store", lambda cli: None)
    target = tmp_path / "out.md"

    _cli(db, sid).save_conversation(f"/save md {target} --concise")

    assert target.read_text(encoding="utf-8") == rpc(session_id=sid, format="md", concise=True)["content"]


def test_save_concise_flag_is_not_taken_for_a_filename_and_other_formats_refuse(db, tmp_path, monkeypatch, capsys):
    sid = _chat(db)
    monkeypatch.setattr("hermes_cli.gateway_tools_notify.gateway_store", lambda cli: None)
    monkeypatch.setattr("hermes_cli.cli_session_mixin.get_hermes_home", lambda: tmp_path)

    _cli(db, sid).save_conversation("/save md --concise")
    _cli(db, sid).save_conversation("/save json --concise")

    saved = list((tmp_path / "sessions" / "saved").glob("*"))
    assert len(saved) == 1 and saved[0].suffix == ".md" and "--concise" not in saved[0].name
    assert "only applies to Markdown" in capsys.readouterr().out


def test_a_legacy_save_with_trailing_words_still_saves(db, tmp_path, monkeypatch):
    sid = _chat(db)
    monkeypatch.setattr("hermes_cli.gateway_tools_notify.gateway_store", lambda cli: None)
    target = tmp_path / "old.md"

    _cli(db, sid).save_conversation(f"/save md {target} some trailing words")

    assert "TOOL-OUTPUT" in target.read_text(encoding="utf-8")


def test_save_without_the_flag_is_unchanged(db, tmp_path, monkeypatch):
    sid = _chat(db)
    monkeypatch.setattr("hermes_cli.gateway_tools_notify.gateway_store", lambda cli: None)
    target = tmp_path / "notes.md"

    _cli(db, sid).save_conversation(f"/save md {target}")

    assert "TOOL-OUTPUT" in target.read_text(encoding="utf-8")


def test_copy_all_copies_the_stored_whole_conversation_not_the_live_context(db, monkeypatch):
    sid = _chat(db)
    monkeypatch.setattr("hermes_cli.gateway_tools_notify.gateway_store", lambda cli: None)
    cli = _cli(db, sid)

    with patch("hermes_cli.clipboard.write_clipboard_text", return_value=True) as clip, \
            patch("hermes_cli.clipboard.is_remote_shell_session", return_value=False):
        cli.process_command("/copy all")

    (text,), _ = clip.call_args
    assert "question 0" in text and "last answer" in text and "only the live tail" not in text
    assert "TOOL-OUTPUT" not in text


def test_copy_all_over_a_remote_shell_uses_the_terminal_clipboard_and_never_truncates(db, monkeypatch, capsys):
    sid = _chat(db)
    monkeypatch.setattr("hermes_cli.gateway_tools_notify.gateway_store", lambda cli: None)
    cli = _cli(db, sid)
    sent = []
    cli._write_osc52_clipboard = sent.append

    with patch("hermes_cli.clipboard.write_clipboard_text") as native, \
            patch("hermes_cli.clipboard.is_remote_shell_session", return_value=True):
        cli.process_command("/copy all")
        assert len(sent) == 1 and "answer 2" in sent[0] and not native.called

        monkeypatch.setattr("hermes_cli.session_export.OSC52_MAX_TEXT_BYTES", 10)
        sent.clear()
        cli.process_command("/copy all")

    assert sent == []  # too large for OSC 52: nothing is sent, and it says so instead of cutting the text


def test_existing_copy_forms_are_untouched(db, monkeypatch):
    sid = _chat(db)
    cli = _cli(db, sid)
    cli.conversation_history = [{"role": "assistant", "content": "one"}, {"role": "assistant", "content": "two"}]

    with patch("hermes_cli.clipboard.write_clipboard_text", return_value=True) as clip, \
            patch("hermes_cli.clipboard.is_remote_shell_session", return_value=False):
        cli.process_command("/copy")
        cli.process_command("/copy 1")

    assert [c.args[0] for c in clip.call_args_list] == ["two", "one"]


def test_gateway_owned_session_asks_the_owner_and_an_old_owner_is_reported_not_papered_over(db, monkeypatch, capsys):
    from hermes_cli.gateway_client import GatewayRPCError

    calls = []

    class Conn:
        def call(self, method, **params):
            calls.append((method, params))
            raise GatewayRPCError("invalid_params")

    monkeypatch.setattr("hermes_cli.gateway_tools_notify.gateway_store", lambda cli: SimpleNamespace(connection=Conn()))
    cli = _cli(db, "s")

    with patch("hermes_cli.clipboard.write_clipboard_text") as clip:
        cli.process_command("/copy all")

    assert calls == [("session.export", {"session_id": "s", "format": "md", "redact": False, "concise": True})]
    assert not clip.called  # nothing is copied from the rows loaded in this process
    assert "too old" in capsys.readouterr().out
