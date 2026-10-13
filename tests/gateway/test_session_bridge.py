"""Unit tests for the session bridge (patch 0005): routing/mapping logic.

Covers the pure functions (source gate, row classification, delta detection, event
shapes, input routing / loop guard, protocol status payloads) and the state-file
round-trip the adapter's is_bridge_room hook reads.
"""


import time

from gateway.session_bridge import (
    AGENT_STATUS_KEY,
    BridgeState,
    SessionThread,
    detect_turn_delta,
    error_reply_rows,
    final_reply_rows,
    is_bridge_room,
    result_line,
    root_announcement,
    route_inbound_event,
    settled_status,
    should_mirror_session,
    status_edit_content,
    thread_reply_content,
    user_message_body,
    visible_user_rows,
    working_status,
)

OWN = "@fbhermes:dsflx.wtf"
ROOT = "$root1"


def _row(id_, role, content="", **kw):
    return {"id": id_, "role": role, "content": content, "tool_calls": kw.get("tool_calls", ""),
            "finish_reason": kw.get("finish_reason", ""), "_compressed_summary": kw.get("summary", 0),
            "message_uid": kw.get("uid", f"u{id_}")}


# ------------------------------------------------------------------ result headline

def test_result_line_first_line_only():
    assert result_line("**Done.** details follow\nmore") == "**Done.** details follow"
    assert result_line("**Done.**") == "Done."      # bold-wrapped lead-in unwrapped (gateway parity)
    assert result_line("# Headed\nbody") == "Headed"
    assert result_line("\n\nsecond line") == "second line"
    assert result_line("") == ""
    assert result_line("x" * 300)[:240] == "x" * 240


# ------------------------------------------------------------------ session gate

def _sess(**over):
    base = {"source": "desktop", "message_count": 5, "started_at": time.time() - 60,
            "last_activity_at": time.time(), "archived": 0, "hidden": 0}
    base.update(over)
    return base


def test_should_mirror_source_and_content_rules():
    now = time.time()
    assert should_mirror_session(_sess(), ("desktop", "tui", "webui"), backfill_days=7, now=now)
    assert not should_mirror_session(_sess(source="matrix"), ("desktop", "tui", "webui"),
                                     backfill_days=7, now=now)
    assert not should_mirror_session(_sess(source="kanban"), ("desktop", "tui", "webui", "kanban"),
                                     backfill_days=7, now=now)
    assert not should_mirror_session(_sess(message_count=0), ("desktop",), backfill_days=7, now=now)
    assert not should_mirror_session(_sess(archived=1), ("desktop",), backfill_days=7, now=now)
    assert should_mirror_session(_sess(source="DESKTOP"), ("desktop",), backfill_days=7,
                                 now=now)  # case-insensitive source match


def test_should_mirror_backfill_window():
    now = time.time()
    old = _sess(started_at=now - 30 * 86400, last_activity_at=now - 30 * 86400)
    assert not should_mirror_session(old, ("desktop",), backfill_days=7, now=now)
    # A freshly-active old session still mirrors (sidebar-parity: activity clock, not birth).
    assert should_mirror_session(_sess(started_at=now - 30 * 86400), ("desktop",),
                                 backfill_days=7, now=now)


# ------------------------------------------------------------------ row classification

def test_visible_and_final_row_classification():
    rows = [
        _row(1, "user", "hi"),
        _row(2, "user", "   "),           # whitespace: not visible
        _row(3, "assistant", "", tool_calls="[{}]"),  # tool-call carrier: not final
        _row(4, "assistant", "answer"),   # final
        _row(5, "assistant", "summary", summary=1),   # compression summary: not final
        _row(6, "tool", "result"),
        _row(7, "assistant", "boom", finish_reason="agent_error"),
    ]
    assert [r["id"] for r in visible_user_rows(rows)] == [1]
    assert [r["id"] for r in final_reply_rows(rows)] == [4, 7]
    assert [r["id"] for r in error_reply_rows(rows)] == [7]


def test_detect_turn_delta_cursor():
    rows = [_row(1, "user", "old"), _row(2, "assistant", "old reply"),
            _row(3, "user", "new"), _row(4, "assistant", "new reply")]
    delta = detect_turn_delta(rows, 2)
    assert [r["id"] for r in delta.new_user_rows] == [3]
    assert [r["id"] for r in delta.final_replies] == [4]
    assert detect_turn_delta(rows, 10).new_user_rows == []


# ------------------------------------------------------------------ wire shapes

def test_thread_reply_content_shape():
    content = thread_reply_content("hello", ROOT, agent_status=working_status("r1"))
    assert content["msgtype"] == "m.text"
    assert content["m.relates_to"] == {"rel_type": "m.thread", "event_id": ROOT,
                                       "is_falling_back": True, "m.in_reply_to": {"event_id": ROOT}}
    assert content[AGENT_STATUS_KEY] == {"v": 1, "state": "working", "run": "r1"}
    assert "m.relates_to" not in thread_reply_content("root body", "")


def test_status_edit_content_shape():
    status = settled_status("r1", "**Done.** it worked", model="m1", cost_usd=0.0123456789)
    edit = status_edit_content("Done. it worked", "$bubble", status)
    assert edit["m.relates_to"] == {"rel_type": "m.replace", "event_id": "$bubble"}
    assert edit["m.new_content"][AGENT_STATUS_KEY]["state"] == "done"
    # Non-integral cost never goes on the wire as a float (canonical JSON rejects it);
    # it rides the result headline instead.
    assert "cost_usd" not in edit[AGENT_STATUS_KEY]
    assert edit[AGENT_STATUS_KEY]["result"] == "**Done.** it worked ($0.01)"
    assert edit[AGENT_STATUS_KEY]["model"] == "m1"
    integral = settled_status("r1", "ok", cost_usd=2.0)
    assert integral["cost_usd"] == 2 and isinstance(integral["cost_usd"], int)
    assert integral["result"] == "ok"


def test_settled_status_failed_and_no_cost():
    status = settled_status("r1", "", failed=True)
    assert status["state"] == "failed" and "result" not in status and "cost_usd" not in status
    zero = settled_status("r1", "done thing", cost_usd=0.0)
    assert zero["cost_usd"] == 0  # integral zero still rides the protocol field


def test_root_announcement_records_source():
    body = root_announcement("My task", "desktop", 1791846753)
    assert "My task" in body and "desktop" in body


def test_user_message_body_attribution():
    assert user_message_body("derek", "tui", "do it").startswith("**derek (tui):**")


# ------------------------------------------------------------------ input routing / loop guard

ROOTS = {ROOT: "20261012_abc"}


def _content(**over):
    base = {"msgtype": "m.text", "body": "steer", "m.relates_to": {
        "rel_type": "m.thread", "event_id": ROOT}}
    base.update(over)
    return base


def test_routes_known_thread_reply():
    assert route_inbound_event("@phrac:dsflx.wtf", OWN, _content(), ROOTS) == "20261012_abc"


def test_loop_guard_rejects_everything_else():
    assert route_inbound_event(OWN, OWN, _content(), ROOTS) is None                # self-sent
    assert route_inbound_event("", OWN, _content(), ROOTS) is None                 # no sender
    edit = _content(**{"m.relates_to": {"rel_type": "m.replace", "event_id": ROOT}})
    assert route_inbound_event("@phrac:dsflx.wtf", OWN, edit, ROOTS) is None       # edits never steer
    assert route_inbound_event("@x:y", OWN, _content(msgtype="m.notice"), ROOTS) is None
    assert route_inbound_event("@x:y", OWN, _content(**{AGENT_STATUS_KEY: {"v": 1}}), ROOTS) is None
    assert route_inbound_event("@x:y", OWN, {"msgtype": "m.text", "body": "hi"}, ROOTS) is None  # room chatter
    unknown = _content(**{"m.relates_to": {"rel_type": "m.thread", "event_id": "$other"}})
    assert route_inbound_event("@x:y", OWN, unknown, ROOTS) is None                # unknown thread


def test_route_is_sender_case_insensitive_on_own_mxid():
    assert route_inbound_event("@FBHERMES:dsflx.wtf", OWN, _content(), ROOTS) is None


def test_controller_gate():
    controllers = frozenset({"@phrac:dsflx.wtf"})
    assert route_inbound_event("@PHRAC:dsflx.wtf", OWN, _content(), ROOTS, controllers) == "20261012_abc"
    assert route_inbound_event("@rando:dsflx.wtf", OWN, _content(), ROOTS, controllers) is None
    # empty controllers → anyone but self may steer
    assert route_inbound_event("@rando:dsflx.wtf", OWN, _content(), ROOTS, frozenset()) == "20261012_abc"


# ------------------------------------------------------------------ state file + adapter hook

def test_bridge_state_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "session_bridge.json"
    st = BridgeState(path)
    st.room_id = "!room:dsflx.wtf"
    st.sessions["s1"] = SessionThread(root="$r", last_msg_id=42, source="desktop",
                                      title="t", profile="default",
                                      open={"r1": {"bubble": "$b", "opened_at": 1.0, "user_row": 3}})
    st.save()
    reloaded = BridgeState(path)
    assert reloaded.room_id == "!room:dsflx.wtf"
    assert reloaded.sessions["s1"].last_msg_id == 42
    assert reloaded.sessions["s1"].open["r1"]["bubble"] == "$b"
    monkeypatch.setattr("gateway.session_bridge._state_path", lambda: path)
    assert is_bridge_room("!room:dsflx.wtf")
    assert not is_bridge_room("!other:dsflx.wtf")
    assert not is_bridge_room("")


def test_is_bridge_room_without_state_file(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.session_bridge._state_path",
                        lambda: tmp_path / "missing" / "session_bridge.json")
    assert not is_bridge_room("!room:dsflx.wtf")


def test_bridge_config_absent_and_present():
    from gateway.session_bridge import bridge_config

    class PC:
        extra: dict = {}

    assert bridge_config(PC()) is None
    PC.extra = {"session_bridge": True, "session_bridge_room": "!r:d.wtf",
                "session_bridge_sources": "desktop, TUI ", "session_bridge_invite": "@phrac:d.wtf",
                "session_bridge_controllers": "@phrac:d.wtf,@mom:d.wtf"}
    cfg = bridge_config(PC())
    assert cfg["room_id"] == "!r:d.wtf"
    assert cfg["sources"] == ("desktop", "tui")
    assert cfg["invite"] == ("@phrac:d.wtf",)
    assert cfg["controllers"] == ("@phrac:d.wtf", "@mom:d.wtf")
