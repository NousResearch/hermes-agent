"""``hermes chat -c <chat> -Q`` into a Bot Chat a live Desktop/TUI holds delivers through that owner.

Bot DMs and cron's ``deliver: bot-chat`` already hand a message to the process that holds the Bot
Chat open (``tools.bot_live_delivery``) and read the reply back; the CLI one-shot was the only
producer refused with SESSION_NOT_OWNED instead. These tests drive the real ``_run_single_query_mode``
entry with a lease acquired the way the Desktop backend acquires it.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import cli
from hermes_cli.active_sessions import try_acquire_active_session
from hermes_state import SessionDB
from tools import bot_live_delivery as mailbox


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_hermes_home", tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    db.create_session(session_id="other", source="cli")
    db.set_session_title("other", "Scratch")
    db.close()
    return tmp_path


def _desktop_owner(home, *, consumer: bool = True):
    lease, refusal = try_acquire_active_session(
        session_id="chat", surface="desktop", config={}, registry_home=home,
        metadata=dict(live_session_id="live-chat", bot_live_delivery_consumer=consumer))
    assert refusal is None
    return lease


def _cli(session_id: str) -> SimpleNamespace:
    claims: list = []

    def _claim(surface="cli", *, stderr=False):
        claims.append(surface)
        return False  # what a real claim against a Desktop-held chat does

    return SimpleNamespace(session_id=session_id, _resumed=True, _claim_active_session=_claim, _claims=claims)


def _capture_deliveries(monkeypatch) -> list:
    """Record what the CLI hands to the mailbox (the admission record, incl. author and id)."""
    from hermes_cli import quiet_single_query as qsq

    seen: list = []
    real = mailbox.deliver_to_live_owner

    def _wrapped(home, owner, message, **kwargs):
        record = real(home, owner, message, **kwargs)
        seen.append(record)
        return record

    monkeypatch.setattr(mailbox, "deliver_to_live_owner", _wrapped)
    return seen


def _answer_as_owner(home, owner, reply: str) -> threading.Thread:
    """The Desktop side: claim the queued delivery and settle it with *reply*."""
    def _run():
        for _ in range(200):
            claimed = mailbox.claim_pending_delivery(home, owner)
            if claimed:
                mailbox.complete_delivery(home, claimed["delivery_id"], status="settled", reply=reply)
                return
            threading.Event().wait(0.02)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def test_a_quiet_turn_into_the_desktop_held_bot_chat_is_answered_by_the_owner(home, capsys, monkeypatch):
    lease = _desktop_owner(home)
    try:
        owner = mailbox.find_canonical_live_owner(home)
        assert owner is not None
        monkeypatch.setenv("HERMES_SESSION_SOURCE", "claude-code")
        seen = _capture_deliveries(monkeypatch)
        fake = _cli("chat")
        thread = _answer_as_owner(home, owner, "Option A. Live.")
        with pytest.raises(SystemExit) as exc:
            cli._run_single_query_mode(fake, "new architecture page is up", None, True, True)
        thread.join(5)
        assert exc.value.code == 0
        out, err = capsys.readouterr()
        assert out.strip() == "Option A. Live."
        assert "session_id: chat" in err
        assert "open in another Hermes window" not in err
        assert fake._claims == []  # never contended the lease
        assert [r["message"] for r in seen] == ["new architecture page is up"]
        assert seen[0]["author"] == {"id": "cli:claude-code", "name": "claude-code", "is_bot": False}
        assert mailbox.read_delivery_result(home, seen[0]["delivery_id"])["status"] == "settled"  # receipt retained
    finally:
        lease.release()


def test_a_turn_report_records_the_owner_delivered_outcome(home, tmp_path, monkeypatch):
    from hermes_cli.quiet_single_query import TURN_REPORT_FILE_ENV, read_turn_report
    import os

    lease = _desktop_owner(home)
    try:
        report = tmp_path / "turn.json"
        monkeypatch.setenv(TURN_REPORT_FILE_ENV, str(report))
        thread = _answer_as_owner(home, mailbox.find_canonical_live_owner(home), "done")
        with pytest.raises(SystemExit) as exc:
            cli._run_single_query_mode(_cli("chat"), "ping", None, True, True)
        thread.join(5)
        assert exc.value.code == 0
        # ``reply`` joined the report contract in write_turn_report (it is what the run printed,
        # so a spawner can relay it); the owner's answer must reach it, not just stdout.
        assert read_turn_report(str(report), os.getpid()) == {
            "pid": os.getpid(), "exit_code": 0, "error": "", "reply": "done"}
        assert TURN_REPORT_FILE_ENV not in os.environ  # popped, as the normal turn does
    finally:
        lease.release()


def test_a_failed_owner_turn_exits_nonzero_with_the_owner_error(home, capsys):
    lease = _desktop_owner(home)
    try:
        owner = mailbox.find_canonical_live_owner(home)

        def _fail():
            for _ in range(200):
                claimed = mailbox.claim_pending_delivery(home, owner)
                if claimed:
                    mailbox.complete_delivery(home, claimed["delivery_id"], status="failed", error="model refused")
                    return
                threading.Event().wait(0.02)

        thread = threading.Thread(target=_fail, daemon=True)
        thread.start()
        with pytest.raises(SystemExit) as exc:
            cli._run_single_query_mode(_cli("chat"), "ping", None, True, True)
        thread.join(5)
        assert exc.value.code == 1
        assert "model refused" in capsys.readouterr().err
    finally:
        lease.release()


def test_an_unanswered_delivery_is_reported_as_queued_and_never_resent(home, capsys, monkeypatch):
    from hermes_cli import quiet_single_query as qsq

    lease = _desktop_owner(home)
    try:
        monkeypatch.setattr(qsq, "LIVE_OWNER_WAIT_SECONDS", 0.2)
        seen = _capture_deliveries(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            cli._run_single_query_mode(_cli("chat"), "ping", None, True, True)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Do not resend" in err and seen[0]["delivery_id"] in err
        assert mailbox.read_delivery_result(home, seen[0]["delivery_id"])["status"] == "queued"  # still the owner's to take
    finally:
        lease.release()


@pytest.mark.parametrize("case", ["other_session", "no_owner", "owner_without_mailbox"])
def test_every_other_target_keeps_the_claim_or_refuse_path(home, case, capsys):
    lease = None
    session_id = "chat"
    if case == "other_session":
        lease = _desktop_owner(home)
        session_id = "other"
    elif case == "owner_without_mailbox":
        lease = _desktop_owner(home, consumer=False)
    try:
        fake = _cli(session_id)
        with pytest.raises(SystemExit) as exc:
            cli._run_single_query_mode(fake, "ping", None, True, True)
        assert exc.value.code == 1
        assert fake._claims == ["cli"]  # the ordinary claim ran (and, in this harness, refused)
        assert not mailbox.has_mailbox(home)  # nothing was admitted to the owner's mailbox
    finally:
        if lease is not None:
            lease.release()


def test_a_fresh_session_or_an_image_or_stream_json_never_takes_the_lane(home):
    lease = _desktop_owner(home)
    try:
        for kwargs in ({"image": "/tmp/x.png"}, {"stream_json": True}):
            fake = _cli("chat")
            with pytest.raises(SystemExit):
                cli._run_single_query_mode(fake, "ping", kwargs.get("image"), True, True,
                                           stream_json=kwargs.get("stream_json", False))
            assert fake._claims == ["cli"]
        fresh = _cli("chat")
        fresh._resumed = False  # a new session id, no -c: nothing a live owner could hold
        with pytest.raises(SystemExit):
            cli._run_single_query_mode(fresh, "ping", None, True, True)
        assert fresh._claims == ["cli"]
    finally:
        lease.release()
