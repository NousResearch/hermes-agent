"""``hermes chat -c <chat> -Q`` into the Bot Chat the profile authority serves delivers through it.

Bot DMs and cron's ``deliver: bot-chat`` already hand a message to the authority that serves the
Bot Chat (``tools.bot_live_delivery``) and read the reply back; the CLI one-shot was the only
producer refused with SESSION_NOT_OWNED instead. These tests drive the real
``_run_single_query_mode`` entry with the authority's discovery and delivery doors stubbed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli
from tools import bot_live_delivery as mailbox

_OWNER = {"session_id": "chat", "canonical": True, "profile_home": "", "lease_id": "i1", "live_session_id": "chat"}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_hermes_home", tmp_path)
    return tmp_path


def _cli(session_id: str) -> SimpleNamespace:
    claims: list = []

    def _claim(surface="cli", *, stderr=False):
        claims.append(surface)
        return False  # what a real claim against an authority-held chat does

    return SimpleNamespace(session_id=session_id, _resumed=True, _claim_active_session=_claim, _claims=claims)


def _authority(monkeypatch, *, settles: dict | None = None, ready: bool = True) -> list:
    """Stub the authority: discovery answers the Bot Chat ``chat``, the deliver door admits, and
    the receipt settles as *settles* (None: still queued when the wait ends). Returns admissions."""
    admitted: list = []

    def _find(home):
        if not ready:
            raise ValueError("profile authority is not ready")
        return dict(_OWNER, profile_home=str(home))

    def _deliver(home, owner, message, *, author=None, **_kw):
        admitted.append({"message": message, "author": author, "delivery_id": "d1"})
        return {"status": "queued", "delivery_id": "d1", "message": message}

    def _await(home, delivery_id, timeout, **_kw):
        return dict(settles, delivery_id=delivery_id) if settles else {"status": "queued", "delivery_id": delivery_id}

    monkeypatch.setattr(mailbox, "find_canonical_live_owner", _find)
    monkeypatch.setattr(mailbox, "deliver_to_live_owner", _deliver)
    monkeypatch.setattr(mailbox, "await_delivery", _await)
    return admitted


def test_a_quiet_turn_into_the_bot_chat_is_answered_by_the_authority(home, capsys, monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "claude-code")
    admitted = _authority(monkeypatch, settles={"status": "settled", "reply": "Option A. Live."})
    fake = _cli("chat")
    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(fake, "new architecture page is up", None, True, True)
    assert exc.value.code == 0
    out, err = capsys.readouterr()
    assert out.strip() == "Option A. Live."
    assert "session_id: chat" in err
    assert fake._claims == []  # never contended the lease
    assert [a["message"] for a in admitted] == ["new architecture page is up"]
    assert admitted[0]["author"] == {"id": "cli:claude-code", "name": "claude-code", "is_bot": False}


def test_a_turn_report_records_the_owner_delivered_outcome(home, tmp_path, monkeypatch):
    from hermes_cli.quiet_single_query import TURN_REPORT_FILE_ENV, read_turn_report
    import os

    _authority(monkeypatch, settles={"status": "settled", "reply": "done"})
    report = tmp_path / "turn.json"
    monkeypatch.setenv(TURN_REPORT_FILE_ENV, str(report))
    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(_cli("chat"), "ping", None, True, True)
    assert exc.value.code == 0
    # ``reply`` joined the report contract in write_turn_report (it is what the run printed,
    # so a spawner can relay it); the owner's answer must reach it, not just stdout.
    assert read_turn_report(str(report), os.getpid()) == {
        "pid": os.getpid(), "exit_code": 0, "error": "", "reply": "done"}
    assert TURN_REPORT_FILE_ENV not in os.environ  # popped, as the normal turn does


def test_a_failed_owner_turn_exits_nonzero_with_the_owner_error(home, capsys, monkeypatch):
    _authority(monkeypatch, settles={"status": "failed", "error": "model refused"})
    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(_cli("chat"), "ping", None, True, True)
    assert exc.value.code == 1
    assert "model refused" in capsys.readouterr().err


def test_an_unanswered_delivery_is_reported_as_queued_and_never_resent(home, capsys, monkeypatch):
    admitted = _authority(monkeypatch, settles=None)
    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(_cli("chat"), "ping", None, True, True)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Do not resend" in err and admitted[0]["delivery_id"] in err


@pytest.mark.parametrize("case", ["other_session", "authority_not_ready"])
def test_every_other_target_keeps_the_claim_or_refuse_path(home, case, monkeypatch):
    admitted = _authority(monkeypatch, settles={"status": "settled", "reply": "x"}, ready=case != "authority_not_ready")
    fake = _cli("other" if case == "other_session" else "chat")
    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(fake, "ping", None, True, True)
    assert exc.value.code == 1
    assert fake._claims == ["cli"]  # the ordinary claim ran (and, in this harness, refused)
    assert admitted == []  # nothing was handed to the authority


def test_a_fresh_session_or_an_image_or_stream_json_never_takes_the_lane(home, monkeypatch):
    admitted = _authority(monkeypatch, settles={"status": "settled", "reply": "x"})
    for kwargs in ({"image": "image.png"}, {"stream_json": True}):
        fake = _cli("chat")
        with pytest.raises(SystemExit):
            cli._run_single_query_mode(fake, "ping", kwargs.get("image"), True, True,
                                       stream_json=kwargs.get("stream_json", False))
        assert fake._claims == ["cli"]
    fresh = _cli("chat")
    fresh._resumed = False  # a new session id, no -c: nothing the authority could hold
    with pytest.raises(SystemExit):
        cli._run_single_query_mode(fresh, "ping", None, True, True)
    assert fresh._claims == ["cli"]
    assert admitted == []
