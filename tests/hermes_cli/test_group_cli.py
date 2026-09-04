"""Behavior contract for ``hermes group`` (hosted-room transport).

The CLI relays a message into a gateway-hosted Group Chat AS THE USER from
any session, and ``send --wait`` follows the typed room log until the
discussion settles. Tests drive the room log by hand (no worker), mirroring
``tests/tui_gateway/test_hosted_room_service.py``.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import sys

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from hermes_cli.subcommands import group as g


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    for profile in ("pax", "archie"):
        (root / "profiles" / profile).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    return root


def _db():
    return hosted_rooms.default_db_path()


def mk_room(room_id="room-1", name="DevTeam", authority=None):
    return hosted_rooms.create_room(
        _db(),
        room_id=room_id,
        name=name,
        members=[
            {"member_id": "pax", "profile": "pax", "handle": "pax"},
            {"member_id": "archie", "profile": "archie", "handle": "archie"},
        ],
        authority_gateway_id=authority or hosted_rooms.local_authority_gateway_id(),
    )


def _run(argv):
    return g.cmd_group(g.build_args(argv))


def _out_json(capsys):
    return json.loads(capsys.readouterr().out)


# ── resolution + list ────────────────────────────────────────────────────────


def test_resolve_exact_ci_id_and_missing(home):
    mk_room()
    assert g.resolve_room("DevTeam").room_id == "room-1"
    assert g.resolve_room("devteam").room_id == "room-1"
    assert g.resolve_room("room-1").room_id == "room-1"
    assert g.resolve_room("DevTeam").kind == "hosted"
    with pytest.raises(g.GroupCLIError, match="No group named"):
        g.resolve_room("nope")
    with pytest.raises(g.GroupCLIError, match="required"):
        g.resolve_room("  ")


def test_resolve_ambiguous_ci_fails_closed_but_exact_wins(home):
    mk_room("room-1", "DevTeam")
    mk_room("room-2", "devteam")
    with pytest.raises(g.GroupCLIError, match="ambiguous"):
        g.resolve_room("DEVTEAM")
    assert g.resolve_room("DevTeam").room_id == "room-1"


def test_list_json_and_plain(home, capsys):
    mk_room()
    mk_room("room-2", "Remote", authority="install:elsewhere")
    assert _run(["list", "--json"]) == 0
    rows = _out_json(capsys)
    by_id = {r["room_id"]: r for r in rows}
    assert by_id["room-1"] == {
        "kind": "hosted",
        "name": "DevTeam",
        "room_id": "room-1",
        "members": ["pax", "archie"],
        "managed_here": True,
    }
    assert by_id["room-2"]["managed_here"] is False
    assert _run(["list"]) == 0
    out = capsys.readouterr().out
    assert "DevTeam  [hosted] (room-1)  members: pax, archie" in out
    assert "[managed elsewhere]" in out


def test_list_empty(home, capsys):
    assert _run(["list"]) == 0
    assert "No groups found" in capsys.readouterr().out


# ── create ───────────────────────────────────────────────────────────────────


def test_create_hosted(home, capsys):
    rc = _run(["create", "DevTeam", "--member", "pax", "--member", "archie", "--json"])
    assert rc == 0
    out = _out_json(capsys)
    assert out["name"] == "DevTeam"
    assert out["room_id"].startswith("grp-")
    assert out["authority_gateway_id"] == hosted_rooms.local_authority_gateway_id()
    assert [m["handle"] for m in out["members"]] == ["pax", "archie"]
    # The new room is now resolvable and driven here.
    assert g.resolve_room("DevTeam").managed_here is True


def test_create_default_profile_handle_is_hermes(home, capsys):
    assert _run(["create", "X", "--member", "default", "--member", "pax", "--json"]) == 0
    assert [m["handle"] for m in _out_json(capsys)["members"]] == ["hermes", "pax"]


def test_create_rejects_bad_roster(home, capsys):
    assert _run(["create", "X", "--member", "pax"]) == 1
    assert "between 2 and 6" in capsys.readouterr().err
    assert _run(["create", "X", "--member", "pax", "--member", "ghost"]) == 1
    assert "ghost" in capsys.readouterr().err


# ── send ─────────────────────────────────────────────────────────────────────


def test_send_appends_user_event_with_relay_attribution(home, capsys, monkeypatch):
    mk_room()
    monkeypatch.setenv("HERMES_PROFILE", "pax")
    rc = _run(["send", "DevTeam", "plan the release", "--as", "Pax via Discord", "--json"])
    assert rc == 0
    out = _out_json(capsys)
    assert out["kind"] == "hosted" and out["room_id"] == "room-1"
    assert out["message_id"].startswith("user:")
    event = out["raw"]
    assert event["kind"] == "message.user"
    assert event["actor"] == {
        "kind": "user",
        "id": "cli",
        "profile": "pax",
        "display_name": "Pax via Discord",
    }
    assert event["payload"] == {"text": "plan the release", "thread_id": "cli"}
    # Landed in the durable log with authority fencing intact.
    logged = hosted_rooms.read_events(_db(), room_id="room-1", since_seq=0)["events"]
    assert [e["kind"] for e in logged] == ["message.user"]
    assert logged[0]["authority_epoch"] == 1


def test_send_default_label_and_thread_sanitizing(home, capsys):
    mk_room()
    assert _run(["send", "DevTeam", "hi", "--thread", "discord_2026_09_03 #x", "--json"]) == 0
    out = _out_json(capsys)
    assert out["thread"] == "discord_2026_09_03-x"
    assert out["raw"]["actor"]["display_name"] == "default relay"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "cli"),
        ("", "cli"),
        ("__", "t-__"),
        ("-abc-", "abc"),
        (".hidden", "t-.hidden"),
        ("a b/c", "a-b-c"),
        ("ok.thread:1", "ok.thread:1"),
    ],
)
def test_thread_id_grammar(raw, expected):
    assert g.thread_id(raw) == expected
    discussion.validate_user_payload({"text": "x", "thread_id": g.thread_id(raw)})


def test_send_refuses_room_managed_elsewhere(home, capsys):
    mk_room("r2", "Remote", authority="install:elsewhere")
    assert _run(["send", "Remote", "hi"]) == 1
    assert "another gateway" in capsys.readouterr().err
    assert hosted_rooms.read_events(_db(), room_id="r2", since_seq=0)["events"] == []


def test_send_stdin_and_empty(home, capsys, monkeypatch):
    mk_room()
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin"))
    assert _run(["send", "DevTeam", "--json"]) == 0
    assert _out_json(capsys)["raw"]["payload"]["text"] == "from stdin"
    monkeypatch.setattr("sys.stdin", io.StringIO("   "))
    assert _run(["send", "DevTeam"]) == 1
    assert "required" in capsys.readouterr().err


def test_send_oversize_text_is_rejected_before_append(home, capsys):
    mk_room()
    big = "x" * (discussion.MAX_USER_TEXT_BYTES + 1)
    assert _run(["send", "DevTeam", big]) == 1
    assert "too large" in capsys.readouterr().err
    assert hosted_rooms.read_events(_db(), room_id="room-1", since_seq=0)["events"] == []


def test_send_event_id_is_idempotent(home, capsys):
    mk_room()
    assert _run(["send", "DevTeam", "once", "--event-id", "k1", "--json"]) == 0
    first = _out_json(capsys)
    assert _run(["send", "DevTeam", "once", "--event-id", "k1", "--json"]) == 0
    again = _out_json(capsys)
    assert first["seq"] == again["seq"] == 1
    assert _run(["send", "DevTeam", "different", "--event-id", "k1"]) == 1  # conflict fails closed


def test_send_plain_output(home, capsys):
    mk_room()
    assert _run(["send", "DevTeam", "hi"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("sent to DevTeam [hosted] id=user:") and "thread=cli" in out


# ── send --wait / log ────────────────────────────────────────────────────────


def _gateway(room):
    return {"kind": "gateway", "id": str(room["authority_gateway_id"])}


def _authority(room):
    return {
        "authority_gateway_id": str(room["authority_gateway_id"]),
        "authority_epoch": int(room["authority_epoch"]),
    }


def _member_turn(room, *, discussion_id, member_id, text, thread="cli", n=1, passed=False):
    """Append the committed shape: message.member then turn.settled(message_event_id)."""
    coords = {
        "discussion_event_id": discussion_id,
        "member_id": member_id,
        "member_index": 0,
        "round_index": 0,
        "task_id": f"task-{n}",
        "thread_id": thread,
        "turn_id": f"turn-{n}",
    }
    message = None
    if not passed:
        message = hosted_rooms.append_event(
            _db(),
            room_id=room["room_id"],
            event_id=f"msg-{n}",
            kind="message.member",
            actor={"kind": "member", "id": member_id, "profile": member_id},
            payload={**coords, "text": text},
            **_authority(room),
        )
    hosted_rooms.append_event(
        _db(),
        room_id=room["room_id"],
        event_id=f"settled-{n}",
        kind="turn.settled",
        actor=_gateway(room),
        payload={
            **coords,
            "seen_through_seq": 1,
            "message_event_id": message["event_id"] if message else None,
            "passed": passed,
        },
        **_authority(room),
    )
    return message


def _activity(room, *, discussion_id, status="settled", reason="consensus", thread="cli"):
    return hosted_rooms.append_event(
        _db(),
        room_id=room["room_id"],
        event_id=f"dactivity:{discussion_id}:{reason}",
        kind="room.activity",
        actor=_gateway(room),
        payload={
            "status": status,
            "reason_code": reason,
            "thread_id": thread,
            "discussion_event_id": discussion_id,
        },
        **_authority(room),
    )


def _sent(room, text="go", key="k"):
    ref = g.resolve_room(room["room_id"])
    return g.HostedTransport().send(ref, text=text, thread="cli", label="Pax", event_key=key)


def test_wait_streams_committed_replies_and_exits_on_settled(home, capsys):
    room = mk_room()
    sent = _sent(room)
    _member_turn(room, discussion_id=sent.message_id, member_id="archie", text="reply from archie", n=1)
    _member_turn(room, discussion_id=sent.message_id, member_id="pax", text="", n=2, passed=True)
    _activity(room, discussion_id=sent.message_id)
    got = []
    rc, summary = g.HostedTransport().wait(
        sent, timeout=5, poll_seconds=0.01, on_reply=lambda s, t: got.append((s, t))
    )
    assert rc == 0
    assert got == [("@archie", "reply from archie")]
    assert summary["status"] == "settled" and summary["reason"] == "consensus"
    assert [r["handle"] for r in summary["replies"]] == ["archie"]


def test_wait_ignores_uncommitted_member_message(home):
    room = mk_room()
    sent = _sent(room)
    hosted_rooms.append_event(
        _db(),
        room_id="room-1",
        event_id="orphan",
        kind="message.member",
        actor={"kind": "member", "id": "archie", "profile": "archie"},
        payload={
            "discussion_event_id": sent.message_id,
            "member_id": "archie",
            "member_index": 0,
            "round_index": 0,
            "task_id": "t",
            "thread_id": "cli",
            "turn_id": "u",
            "text": "never committed",
        },
        **_authority(room),
    )
    _activity(room, discussion_id=sent.message_id, status="bounded", reason="round-cap")
    got = []
    rc, summary = g.HostedTransport().wait(sent, timeout=5, poll_seconds=0.01, on_reply=lambda *a: got.append(a))
    assert rc == 0 and got == [] and summary["status"] == "bounded"


def test_wait_does_not_stream_replies_from_other_discussions_or_threads(home):
    room = mk_room()
    sent = _sent(room)
    other = g.HostedTransport().send(
        g.resolve_room("room-1"), text="other relay", thread="t2", label="Other", event_key="k-other"
    )
    # A committed reply to the OTHER discussion (different discussion id AND thread).
    _member_turn(room, discussion_id=other.message_id, member_id="archie", text="for t2", thread="t2", n=1)
    # A committed reply whose thread matches ours but belongs to another discussion id.
    _member_turn(room, discussion_id="user:stale", member_id="archie", text="stale", thread="cli", n=2)
    # Ours.
    _member_turn(room, discussion_id=sent.message_id, member_id="archie", text="for us", n=3)
    _activity(room, discussion_id=sent.message_id)
    got = []
    rc, summary = g.HostedTransport().wait(sent, timeout=5, poll_seconds=0.01, on_reply=lambda s, t: got.append(t))
    assert rc == 0 and got == ["for us"]
    assert [r["text"] for r in summary["replies"]] == ["for us"]


def test_wait_rejects_non_positive_poll(home):
    room = mk_room()
    sent = _sent(room)
    for bad in (0, -1):
        with pytest.raises(g.GroupCLIError, match="--poll"):
            g.HostedTransport().wait(sent, timeout=1, poll_seconds=bad, on_reply=lambda *a: None)


def test_wait_only_settles_on_own_discussion(home):
    room = mk_room()
    sent = _sent(room)
    _activity(room, discussion_id="user:someone-else", thread="other")
    rc, summary = g.HostedTransport().wait(sent, timeout=0.05, poll_seconds=0.01, on_reply=lambda *a: None)
    assert rc == 3 and summary["status"] == "timeout"


def test_wait_times_out_with_partial_replies(home):
    room = mk_room()
    sent = _sent(room)
    _member_turn(room, discussion_id=sent.message_id, member_id="archie", text="partial", n=1)
    got = []
    rc, summary = g.HostedTransport().wait(sent, timeout=0.05, poll_seconds=0.01, on_reply=lambda s, t: got.append(t))
    assert rc == 3 and got == ["partial"] and len(summary["replies"]) == 1


def test_wait_exit_4_when_room_stop_supersedes(home):
    """A stop fence is room-wide (policy stopped_through_seq), so exit 4 even for a stop aimed elsewhere."""
    room = mk_room()
    sent = _sent(room)
    hosted_rooms.request_room_stop(
        _db(),
        room_id="room-1",
        cancel_id="stop-1",
        expected_gateway_id=str(room["authority_gateway_id"]),
        expected_epoch=int(room["authority_epoch"]),
    )
    rc, summary = g.HostedTransport().wait(sent, timeout=5, poll_seconds=0.01, on_reply=lambda *a: None)
    assert rc == 4 and summary["status"] == "stopped"


def test_wait_warns_once_when_no_worker(home, capsys, monkeypatch):
    room = mk_room()
    sent = _sent(room)
    monkeypatch.setattr(g, "_WORKER_WARN_AFTER_SECONDS", 0.0)
    rc, _ = g.HostedTransport().wait(sent, timeout=0.05, poll_seconds=0.01, on_reply=lambda *a: None)
    assert rc == 3
    assert capsys.readouterr().err.count("no gateway worker") == 1


def test_send_wait_cli_output_and_exit_codes(home, capsys):
    room = mk_room()
    key = "cli-wait"
    # Pre-seed the log so the CLI's own send (same key → same event) sees a finished discussion.
    sent = _sent(room, key=key)
    _member_turn(room, discussion_id=sent.message_id, member_id="archie", text="done deal", n=1)
    _activity(room, discussion_id=sent.message_id)
    rc = _run(["send", "DevTeam", "go", "--event-id", key, "--as", "Pax", "--wait", "--poll", "0.01", "--timeout", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "@archie: done deal" in out
    assert "[group DevTeam: settled (consensus), 1 replies]" in out

    rc = _run(["send", "DevTeam", "go", "--event-id", key, "--as", "Pax", "--wait", "--poll", "0.01", "--timeout", "5", "--json"])
    assert rc == 0
    summary = _out_json(capsys)
    assert summary["status"] == "settled" and summary["replies"][0]["text"] == "done deal"

    rc = _run(["send", "DevTeam", "again", "--thread", "t2", "--wait", "--poll", "0.01", "--timeout", "0.05"])
    assert rc == 3
    assert "timeout" in capsys.readouterr().err


def test_log_renders_committed_transcript_with_relay_attribution(home, capsys):
    room = mk_room()
    sent = _sent(room)
    _member_turn(room, discussion_id=sent.message_id, member_id="archie", text="ack", n=1)
    assert _run(["log", "DevTeam", "--json"]) == 0
    rows = _out_json(capsys)
    assert [(r["speaker"], r["text"]) for r in rows] == [("User (Pax)", "go"), ("@archie", "ack")]
    assert _run(["log", "DevTeam", "--since", str(sent.seq)]) == 0
    assert capsys.readouterr().out.strip() == "@archie: ack"


# ── parser / dispatch ────────────────────────────────────────────────────────


def test_parser_registers_and_dispatches():
    parser = argparse.ArgumentParser(prog="hermes")
    g.build_group_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["group", "send", "DevTeam", "hi", "--as", "X", "--wait"])
    assert args.func is g.cmd_group
    assert args.group_action == "send" and args.as_label == "X" and args.wait is True
    assert args.timeout == g.DEFAULT_WAIT_TIMEOUT_SECONDS


def test_cmd_group_without_action_is_usage_error(capsys):
    assert g.cmd_group(argparse.Namespace(group_action=None)) == 2
    assert "usage" in capsys.readouterr().err


def test_transport_registry_dedupes_by_kind():
    before = len(g.transports())
    g.register_transport(g.HostedTransport())
    assert len(g.transports()) == before


# ── session-bound thread continuity ──────────────────────────────────────────


def test_hosted_thread_follows_session(home, capsys, monkeypatch):
    mk_room()
    monkeypatch.setenv("HERMES_SESSION_ID", "20260903_1200_abc")
    assert _run(["send", "DevTeam", "first", "--json"]) == 0
    t1 = _out_json(capsys)["thread"]
    assert t1 == g.thread_id("s-20260903_1200_abc")
    assert _run(["send", "DevTeam", "second", "--json"]) == 0
    assert _out_json(capsys)["thread"] == t1
    # A different session gets its own thread.
    monkeypatch.setenv("HERMES_SESSION_ID", "20260903_1300_xyz")
    assert _run(["send", "DevTeam", "other", "--json"]) == 0
    assert _out_json(capsys)["thread"] != t1
    # --new-thread from the first session breaks its binding... to a fresh derived label
    monkeypatch.setenv("HERMES_SESSION_ID", "20260903_1200_abc")
    assert _run(["send", "DevTeam", "restart", "--new-thread", "--json"]) == 0
    assert _out_json(capsys)["thread"] == t1  # hosted label is session-derived, so same label is correct
    # --session overrides the env; --thread overrides everything.
    assert _run(["send", "DevTeam", "x", "--session", "S2", "--json"]) == 0
    assert _out_json(capsys)["thread"] == g.thread_id("s-S2")
    assert _run(["send", "DevTeam", "x", "--thread", "explicit", "--json"]) == 0
    assert _out_json(capsys)["thread"] == "explicit"


def test_no_session_uses_shared_cli_thread(home, capsys, monkeypatch):
    mk_room()
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    assert _run(["send", "DevTeam", "x", "--json"]) == 0
    assert _out_json(capsys)["thread"] == "cli"


def test_bindings_file_is_bounded_and_survives_corruption(home, monkeypatch):
    mk_room()
    ref = g.resolve_room("DevTeam")
    monkeypatch.setattr(g, "_THREAD_BINDINGS_MAX", 3)
    for i in range(5):
        g.bind_thread(ref, f"s{i}", f"t{i}")
    assert g.bound_thread(ref, "s0") is None and g.bound_thread(ref, "s4") == "t4"
    g._bindings_path().write_text("{not json")
    assert g.bound_thread(ref, "s4") is None
    g.bind_thread(ref, "s9", "t9")  # rewrites cleanly
    assert g.bound_thread(ref, "s9") == "t9"
    g.forget_thread(ref, "s9")
    assert g.bound_thread(ref, "s9") is None


@pytest.mark.skipif(
    sys.platform == "win32" or "fork" not in multiprocessing.get_all_start_methods(),
    reason="fork start method required; the flock is a no-op without fcntl anyway",
)
def test_bind_thread_is_safe_under_concurrent_writers(home):
    """Many writers, each binding its own session: no binding may be lost."""
    import multiprocessing as mp

    mk_room()
    ref = g.resolve_room("DevTeam")

    def worker(i, home_path):
        import os as _os

        _os.environ["HERMES_HOME"] = home_path
        from hermes_cli.subcommands import group as gg

        gg.bind_thread(ref, f"sess-{i}", f"t{i}")

    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=worker, args=(i, str(home))) for i in range(24)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(10)
    for i in range(24):
        assert g.bound_thread(ref, f"sess-{i}") == f"t{i}", i
    # No temp files left behind, lock file is harmless.
    leftovers = [p.name for p in g._bindings_path().parent.iterdir() if p.name.startswith(".bindings-")]
    assert leftovers == []


def test_explicit_thread_does_not_rebind_session(home, capsys, monkeypatch):
    mk_room()
    monkeypatch.setenv("HERMES_SESSION_ID", "S")
    assert _run(["send", "DevTeam", "a", "--json"]) == 0
    own = _out_json(capsys)["thread"]
    assert _run(["send", "DevTeam", "b", "--thread", "elsewhere", "--json"]) == 0
    assert _out_json(capsys)["thread"] == "elsewhere"
    assert _run(["send", "DevTeam", "c", "--json"]) == 0
    assert _out_json(capsys)["thread"] == own
