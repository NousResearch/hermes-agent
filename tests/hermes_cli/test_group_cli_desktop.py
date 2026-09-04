"""Behavior contract for the Desktop-room transport of ``hermes group``
(hermes_cli/subcommands/group_desktop.py) — envelope out, JSONL lines in."""

from __future__ import annotations

import json
import textwrap
import threading
import time

import pytest

from gateway import hosted_rooms
from hermes_cli.subcommands import group as g
from hermes_cli.subcommands import group_desktop as gd
from tools import group_relay as gr

PROJECTION = textwrap.dedent(
    """\
    ui_meta:
      hermes-bots:
        title: Pax
      hermes-bots-groups:
        version: 3
        updatedAt: 1788476626302
        rooms:
          id:rmtm1xrg6-f5ljm:
            name: TraderChat
            roomId: rmtm1xrg6-f5ljm
            log:
              - id: a1
                from: {kind: user, name: You}
                text: Good clarification from HL founder
                at: 1788474195411
                thread: tmtm3cx5e-h1wzb
              - id: a2
                from: {kind: member, name: socrates}
                text: This is a genuine confirm limb
                at: 1788474200000
                thread: tmtm3cx5e-h1wzb
              - id: a3
                from: {kind: user, name: You, via: Pax via Discord}
                text: relayed question
                at: 1788474300000
                thread: tmtm9
          name:Legacy Room:
            name: Legacy Room
            log:
              - from: {kind: member, name: argus}
                text: hi
                at: 1
    """
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    (root / "profiles" / "pax").mkdir(parents=True)
    (root / "profile.yaml").write_text(PROJECTION, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    return root


def _run(argv):
    return g.cmd_group(g.build_args(argv))


def _out_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_transport_registered_and_lists_desktop_rooms(home, capsys):
    assert [t.kind for t in g.transports()] == ["hosted", "desktop"]
    assert _run(["list", "--json"]) == 0
    rows = {r["name"]: r for r in _out_json(capsys)}
    assert rows["TraderChat"] == {
        "kind": "desktop",
        "name": "TraderChat",
        "room_id": "rmtm1xrg6-f5ljm",
        "members": ["socrates"],
        "managed_here": True,
    }
    assert rows["Legacy Room"]["room_id"] == "Legacy Room" and rows["Legacy Room"]["members"] == ["argus"]


def test_projection_missing_or_corrupt_is_empty(home):
    (home / "profile.yaml").write_text("ui_meta: [not, a, map]\n")
    assert gd.read_projection(home) == {}
    (home / "profile.yaml").unlink()
    assert gd.read_projection(home) == {}
    assert gd.DesktopTransport().list_rooms() == []


def test_name_collision_across_kinds_requires_kind(home, capsys):
    hosted_rooms.create_room(
        hosted_rooms.default_db_path(), room_id="h1", name="TraderChat",
        members=[{"member_id": "pax", "profile": "pax", "handle": "pax"},
                 {"member_id": "default", "profile": "default", "handle": "hermes"}],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id())
    assert _run(["send", "TraderChat", "hi"]) == 1
    assert "ambiguous" in capsys.readouterr().err
    assert g.resolve_room("TraderChat", kind="desktop").kind == "desktop"
    assert g.resolve_room("TraderChat", kind="hosted").kind == "hosted"
    assert _run(["send", "TraderChat", "hi", "--kind", "desktop", "--json"]) == 0
    assert _out_json(capsys)["kind"] == "desktop"


def test_send_enqueues_envelope_new_thread_by_default(home, capsys, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "pax")
    assert _run(["send", "TraderChat", "what now?", "--as", "Pax via Discord", "--json"]) == 0
    out = _out_json(capsys)
    assert out["kind"] == "desktop" and out["seq"] is None
    env = out["raw"]
    assert env["room_id"] == "rmtm1xrg6-f5ljm" and env["room_name"] == "TraderChat"
    assert env["text"] == "what now?" and env["label"] == "Pax via Discord" and env["from_profile"] == "pax"
    assert env["thread"] is None  # default 'cli' → new Desktop thread
    assert gr.pending_count(home) == 1
    assert _run(["send", "TraderChat", "cont", "--thread", "tmtm3cx5e-h1wzb", "--json"]) == 0
    assert _out_json(capsys)["raw"]["thread"] == "tmtm3cx5e-h1wzb"


def test_send_by_name_only_room_omits_room_id(home, capsys):
    assert _run(["send", "Legacy Room", "x", "--json"]) == 0
    env = _out_json(capsys)["raw"]
    assert env["room_id"] == "" and env["room_name"] == "Legacy Room"


def test_send_fails_fast_when_outbox_is_visibly_undrained(home, capsys, monkeypatch):
    monkeypatch.setattr(gd, "_STALE_OUTBOX_MAX_PENDING", 2)
    for _ in range(2):
        env = gr.enqueue(home, room_id="r", room_name="n", text="t", from_profile="p", label="l")
        path = gr.relay_root(home) / gr.OUTBOX_DIR / f"{env['id']}.json"
        old = time.time() - 600
        import os

        os.utime(path, (old, old))
    assert _run(["send", "TraderChat", "hi"]) == 1
    assert "Desktop does not appear to be open" in capsys.readouterr().err


def _sent(home):
    ref = g.resolve_room("TraderChat", kind="desktop")
    return gd.DesktopTransport().send(ref, text="go", thread="cli", label="Pax", event_key=None)


def test_wait_streams_lines_and_maps_done_statuses(home):
    t = gd.DesktopTransport()
    for status, rc in (("settled", 0), ("capped", 0), ("cancelled", 4), ("timeout", 3)):
        sent = _sent(home)
        gr.append_reply_line(home, sent.message_id, {"kind": "accepted", "thread": "tmtmNEW", "group": "TraderChat"})
        gr.append_reply_line(home, sent.message_id, {"kind": "reply", "member": "socrates", "text": "one", "thread": "tmtmNEW"})
        gr.append_reply_line(home, sent.message_id, {"kind": "done", "status": status, "replies": 1})
        got = []
        code, summary = t.wait(sent, timeout=5, poll_seconds=0.01, on_reply=lambda s, x: got.append((s, x)))
        assert code == rc, status
        assert got == [("@socrates", "one")]
        assert summary["status"] == status and summary["thread"] == "tmtmNEW"
        assert [r["text"] for r in summary["replies"]] == ["one"]


def test_wait_streams_incrementally_while_lines_arrive(home):
    sent = _sent(home)

    def writer():
        time.sleep(0.05)
        gr.append_reply_line(home, sent.message_id, {"kind": "accepted", "thread": "t", "group": "TraderChat"})
        time.sleep(0.05)
        gr.append_reply_line(home, sent.message_id, {"kind": "reply", "member": "argus", "text": "late", "thread": "t"})
        gr.append_reply_line(home, sent.message_id, {"kind": "done", "status": "settled", "replies": 1})

    threading.Thread(target=writer, daemon=True).start()
    got = []
    code, _ = gd.DesktopTransport().wait(sent, timeout=5, poll_seconds=0.01, on_reply=lambda s, x: got.append(x))
    assert code == 0 and got == ["late"]


def test_wait_error_line_raises_clean_error(home):
    sent = _sent(home)
    gr.append_reply_line(home, sent.message_id, {"kind": "error", "reason": "room_not_found", "error": "no such room"})
    with pytest.raises(g.GroupCLIError, match=r"no such room \[reason: room_not_found\]"):
        gd.DesktopTransport().wait(sent, timeout=5, poll_seconds=0.01, on_reply=lambda *a: None)


def test_wait_timeout_and_warning(home, capsys, monkeypatch):
    monkeypatch.setattr(gd, "_DESKTOP_WARN_AFTER_SECONDS", 0.0)
    sent = _sent(home)
    code, summary = gd.DesktopTransport().wait(sent, timeout=0.05, poll_seconds=0.01, on_reply=lambda *a: None)
    assert code == 3 and summary["status"] == "timeout"
    assert capsys.readouterr().err.count("Desktop hasn't picked this up") == 1
    with pytest.raises(g.GroupCLIError, match="--poll"):
        gd.DesktopTransport().wait(sent, timeout=1, poll_seconds=0, on_reply=lambda *a: None)


def test_send_wait_cli_end_to_end(home, capsys):
    # Pre-stage the reply file is impossible before the id exists, so run
    # the CLI in a thread and answer it from the outbox.
    result = {}

    def cli():
        result["rc"] = _run(["send", "TraderChat", "go", "--as", "Pax", "--wait", "--poll", "0.01", "--timeout", "5"])

    worker = threading.Thread(target=cli)
    worker.start()
    deadline = time.time() + 3
    env = None
    while time.time() < deadline and env is None:
        pending = list((gr.relay_root(home) / gr.OUTBOX_DIR).glob("*.json"))
        if pending:
            env = json.loads(pending[0].read_text())
        else:
            time.sleep(0.01)
    assert env is not None and env["label"] == "Pax"
    gr.claim_pending(home)
    gr.append_reply_line(home, env["id"], {"kind": "accepted", "thread": "tmtmX", "group": "TraderChat"})
    gr.append_reply_line(home, env["id"], {"kind": "reply", "member": "socrates", "text": "done deal", "thread": "tmtmX"})
    gr.append_reply_line(home, env["id"], {"kind": "done", "status": "settled", "replies": 1})
    worker.join(timeout=5)
    assert result["rc"] == 0
    out = capsys.readouterr().out
    assert "@socrates: done deal" in out and "[group TraderChat: settled (settled), 1 replies]" in out


def test_log_renders_projection_with_via(home, capsys):
    assert _run(["log", "TraderChat", "--json"]) == 0
    rows = _out_json(capsys)
    assert [(r["speaker"], r["text"]) for r in rows] == [
        ("User (You)", "Good clarification from HL founder"),
        ("@socrates", "This is a genuine confirm limb"),
        ("User (Pax via Discord)", "relayed question"),
    ]
    assert rows[2]["thread"] == "tmtm9"
    assert _run(["log", "TraderChat", "--since", "2"]) == 0
    assert capsys.readouterr().out.strip() == "User (Pax via Discord): relayed question"
