"""``hermes groups``: the product's words over the local gateway's succession and custody calls."""
import argparse

import pytest

# The CLI's main module is imported when the tests load, like in the other CLI tests: its start-up
# resolves every sys.path entry, and an interpreter's entries can sit under ~/.hermes (the uv cache),
# which a test must not touch.
from hermes_cli import main
from hermes_cli.subcommands import groups as group

HOST, HERE, OTHER = "install:" + "a" * 32, "install:" + "b" * 32, "install:" + "c" * 32


def status(state="host_unreachable", targets=(HERE,), **extra):
    return {"state": state, "host": {"install_id": HOST, "name": "Mac mini", "reachable": False, "since": 0},
            "this_install": {"install_id": HERE, "name": "Home VPS", "role": "backup"}, "owner": {"name": "Dana"},
            "backups": [{"install_id": HERE, "name": "Home VPS", "successor": True, "readiness": "caught_up",
                         "behind_by": 0, "last_seen": None, "allowed": True, "designated": True, "kind": "backup",
                         "operator_name": "Dana"},
                        {"install_id": OTHER, "name": "Laptop", "successor": True, "readiness": "behind",
                         "behind_by": 3, "last_seen": None, "allowed": True, "designated": True, "kind": "member",
                         "operator_name": "Dana"}],
            "at_risk": {"count": 0}, "moving": None, "conflict": None, "moved": None, "work": None,
            "actions": [{"action": "continue", "targets": list(targets)}] if targets else [],
            "unavailable_reason": None, "previous_host": None, "unavailable_bots": [], "last_attempt": None, **extra}


@pytest.fixture
def gateway(monkeypatch):
    calls, replies = [], {}

    def call(method, **params):
        calls.append((method, params))
        reply = replies[method]
        return reply(**params) if callable(reply) else reply

    monkeypatch.setattr(group, "_call", call)
    replies["groups.list"] = {"rooms": [{"room_id": "room", "name": "Weekend plans"}]}
    return calls, replies


def test_status_prints_the_offline_host_and_where_the_group_can_continue(gateway, capsys):
    calls, replies = gateway
    replies["groups.succession.status"] = status()
    assert group.cmd_status(argparse.Namespace(group="weekend plans")) == 0
    out = capsys.readouterr().out
    assert "Mac mini is offline" in out and "The group is paused." in out
    assert "Home VPS: full copy, up to date (can continue this group)" in out
    assert "Laptop: full copy, 3 messages behind" in out and "Can continue on: Home VPS" in out
    assert calls[-1] == ("groups.succession.status", {"room_id": "room"})


def test_continue_shows_the_summary_and_continues_here(gateway, capsys):
    calls, replies = gateway
    replies["groups.succession.status"] = status()
    replies["groups.succession.prepare"] = {
        "preview_id": "preview_1", "target": {"install_id": HERE, "name": "Home VPS", "operator_name": "Sam"},
        "owner": {"name": "Dana"}, "behind_by": 0, "at_risk": {"count": 0},
        "work": {"completed": 1, "elsewhere": 0, "unknown": 1, "waiting_for_host": 0},
        "unavailable_bots": [{"member_id": "writer", "name": "Writer"}], "cautions": [{"code": "host_may_be_running"}]}
    replies["groups.succession.promote"] = status(
        "ok", targets=(), work={"completed": 1, "elsewhere": 0, "unknown": 1, "waiting_for_host": 0},
        previous_host={"install_id": HOST, "name": "Mac mini", "offline_since": 0})
    assert group.cmd_continue(argparse.Namespace(group="room", on=None, yes=True)) == 0
    out = capsys.readouterr().out
    assert "Continue “room” on Home VPS?" in out and "Sam will manage this group from Home VPS." in out
    assert "1 Bot runs on Mac mini and stays unavailable until the group moves back there: Writer." in out
    assert "Done. “room” now continues on Home VPS." in out
    assert "Work in progress: 1 unknown, 0 waiting for Mac mini. Unknown work won't run again automatically." in out
    assert ("groups.succession.promote", {"room_id": "room", "target_install_id": HERE, "preview_id": "preview_1",
                                          "confirm": True}) in calls


def test_continue_says_when_another_computer_is_better_placed(gateway, monkeypatch, capsys):
    calls, replies = gateway
    replies["groups.succession.status"] = status(targets=(OTHER, HERE))
    monkeypatch.setattr("builtins.input", lambda prompt: (print(prompt), "n")[1])
    assert group.cmd_continue(argparse.Namespace(group="room", on=None, yes=False)) == 1
    assert "Laptop is better placed to continue this group. Continue here anyway?" in capsys.readouterr().out
    assert not any(method == "groups.succession.prepare" for method, _ in calls)
    assert group.cmd_continue(argparse.Namespace(group="room", on="Laptop", yes=True)) == 1
    assert "Run this on Laptop" in capsys.readouterr().err


def test_keep_and_allow_call_the_gateway(gateway, capsys):
    calls, replies = gateway
    replies["groups.succession.status"] = status("continued_on_two", targets=(), conflict={"hosts": [
        {"install_id": HERE, "name": "Home VPS", "since": 1}, {"install_id": OTHER, "name": "Laptop", "since": 2}]})
    replies["groups.succession.keep"] = status("ok", targets=())
    assert group.cmd_keep(argparse.Namespace(group="room", computer="laptop", yes=True)) == 0
    assert ("groups.succession.keep", {"room_id": "room", "install_id": OTHER}) in calls
    replies["groups.custody.allow"] = {"room_id": "room", "install_id": HERE, "allowed": True, "confirmed": False}
    assert group.cmd_backups(argparse.Namespace(group="room", backups_action="allow", computer="here")) == 0
    assert ("groups.custody.allow", {"room_id": "room", "successor": True}) in calls


def test_errors_use_the_products_words(gateway, capsys):
    from hermes_cli.gateway_client import GatewayClientError

    def refuse(**params):
        raise group.GroupCommandError(group._MESSAGES["not_owner"])

    calls, replies = gateway
    replies["groups.succession.status"] = refuse
    assert group.cmd_status(argparse.Namespace(group="room")) == 1
    assert capsys.readouterr().err.strip() == "Only the group's owner can do that."
    assert GatewayClientError  # the real transport maps its reason codes through _MESSAGES


def automatic(state="ready", mode="majority", **extra):
    return {"mode": mode, "state": state, "standby": {"install_id": HERE, "name": "Home VPS"},
            "voters": [{"install_id": HOST, "name": "Mac mini"}, {"install_id": HERE, "name": "Home VPS"}], **extra}


@pytest.mark.parametrize(("value", "line"), [
    (automatic(), "Keeps running on its own. If Mac mini goes offline, Home VPS takes over within a minute."),
    (automatic(mode="careful"), "If Mac mini goes offline, Home VPS takes over after about 3 minutes."),
    (automatic("not_ready", reason="voters_offline", offline=[{"install_id": OTHER, "name": "Laptop"}]),
     "Right now Laptop is offline. If Mac mini goes offline before then, you'll be asked where to continue."),
    (automatic("unavailable", mode="ask", reason="needs_computers", needed=1),
     "Add 1 more always-on computer to make this automatic."),
    (automatic("off", mode="ask"), "Moves only when you choose.")])
def test_status_prints_whether_the_group_moves_by_itself(gateway, capsys, value, line):
    calls, replies = gateway
    replies["groups.succession.status"] = status("ok", targets=(), automatic=value)
    assert group.cmd_status(argparse.Namespace(group="room")) == 0
    assert line in capsys.readouterr().out


def test_status_says_when_the_host_paused_to_stay_safe(gateway, capsys):
    calls, replies = gateway
    replies["groups.succession.status"] = status("paused", targets=(), automatic=automatic(), paused={
        "reason": "lost_majority", "since": 1, "waiting_for": [{"install_id": HERE, "name": "Home VPS"},
                                                               {"install_id": OTHER, "name": "Laptop"}]})
    assert group.cmd_status(argparse.Namespace(group="room")) == 0
    assert ("Paused to stay safe: Mac mini can't reach Home VPS and Laptop. It resumes as soon as one of them is "
            "back.") in capsys.readouterr().out


def test_status_says_when_a_step_promised_elsewhere_was_never_taken(gateway, capsys):
    calls, replies = gateway
    replies["groups.succession.status"] = status("paused", targets=(), automatic=automatic(), paused={
        "reason": "step_not_taken", "since": 1, "waiting_for": [{"install_id": OTHER, "name": "Laptop"}]})
    assert group.cmd_status(argparse.Namespace(group="room")) == 0
    assert ("Paused to stay safe: its next step was promised to another computer, which never took it. Mac mini "
            "resumes once Laptop confirms nothing else happened, or you can continue it anyway in Hermes "
            "Desktop.") in capsys.readouterr().out


def test_move_hands_the_group_over_from_the_host(gateway, capsys):
    calls, replies = gateway
    replies["groups.succession.status"] = status("ok", targets=(), automatic=automatic())
    replies["groups.succession.move"] = status("moved_away", targets=(), moved={
        "to": {"install_id": HERE, "name": "Home VPS"}, "at": 1, "separate_events": 0, "branch_id": None})
    assert group.cmd_move(argparse.Namespace(group="room", computer="Home VPS", yes=True)) == 0
    assert ("groups.succession.move", {"room_id": "room", "target_install_id": HERE}) in calls
    assert "This group moved to Home VPS." in capsys.readouterr().out


def test_the_full_cli_parser_builds_and_routes_the_groups_family(monkeypatch, tmp_path):
    """The whole ``hermes`` tree builds (a merge that registers ``groups`` twice fails here). It is built
    against an empty home, so no installed plugin or profile changes it."""
    (tmp_path / ".hermes").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    parser, _subparsers = main._build_cli_parser()
    parsed = parser.parse_args(["groups", "status", "Weekend plans"])
    assert parsed.func is group.cmd_status and parsed.group == "Weekend plans"
    parsed = parser.parse_args(["groups", "move", "Weekend plans", "Home VPS", "--yes"])
    assert parsed.func is group.cmd_move and parsed.computer == "Home VPS" and parsed.yes


def preview(**extra):
    return {"preview_id": "preview_1", "target": {"install_id": HERE, "name": "Home VPS", "operator_name": "Dana"},
            "owner": {"name": "Dana"}, "behind_by": 0, "at_risk": {"count": 0},
            "work": {"completed": 0, "elsewhere": 0, "unknown": 0, "waiting_for_host": 0}, "unavailable_bots": [],
            "cautions": [], **extra}


def test_the_summary_counts_missing_messages_apart_from_catching_up_and_names_who_cant_be_reached():
    lines = "\n".join(group.summary_lines("Weekend plans", "Mac mini", preview(
        at_risk={"count": 2}, behind_by=3, cautions=[
            {"code": "voters_unreachable", "names": ["Laptop", "Attic"], "count": 2},
            {"code": "voters_unreachable", "names": [], "count": 1}])))
    assert "Home VPS is missing 2 recent messages. They'll appear if Mac mini comes back." in lines
    assert "Home VPS is catching up 3 messages from another computer." in lines
    assert ("Laptop and Attic can't be reached, so this computer can't confirm Mac mini has stopped. Continue only "
            "if Mac mini is really offline.") in lines
    assert "1 computer can't be reached" in lines


def test_status_never_shows_an_install_id(gateway, capsys):
    calls, replies = gateway
    unnamed = status("continued_on_two", targets=(), conflict={
        "hosts": [{"install_id": HERE, "name": None, "since": 1}, {"install_id": OTHER, "name": None, "since": 2}],
        "running_on": {"install_id": OTHER, "name": None}})
    unnamed["host"]["name"] = None
    for row in unnamed["backups"]:
        row["name"] = None
    replies["groups.succession.status"] = unnamed
    assert group.cmd_status(argparse.Namespace(group="room")) == 0
    out = capsys.readouterr().out
    assert "install:" not in out and "another computer is running the group" in out.lower()


def test_status_says_which_computer_runs_the_group_after_a_split_and_offers_keeping_it_first(gateway, capsys):
    calls, replies = gateway
    replies["groups.succession.status"] = status("continued_on_two", targets=(), conflict={
        "hosts": [{"install_id": HERE, "name": "Home VPS", "since": 1}, {"install_id": OTHER, "name": "Laptop",
                                                                          "since": 2}],
        "running_on": {"install_id": OTHER, "name": "Laptop"}})
    assert group.cmd_status(argparse.Namespace(group="room")) == 0
    out = capsys.readouterr().out
    assert "Laptop is running the group; Home VPS stopped." in out
    assert out.index("Keep going on Laptop") < out.index("Switch to Home VPS")


def test_keep_asks_first_and_says_what_is_kept_apart(gateway, monkeypatch, capsys):
    calls, replies = gateway
    replies["groups.succession.status"] = status("continued_on_two", targets=(), conflict={
        "hosts": [{"install_id": HERE, "name": "Home VPS", "since": 1}, {"install_id": OTHER, "name": "Laptop",
                                                                          "since": 2}],
        "running_on": {"install_id": OTHER, "name": "Laptop"}})
    monkeypatch.setattr("builtins.input", lambda prompt: (print(prompt), "n")[1])
    assert group.cmd_keep(argparse.Namespace(group="room", computer="Home VPS", yes=False)) == 1
    assert ("Switch “room” to Home VPS? It continues there, and messages written only on Laptop while the two were "
            "apart are kept separately, not mixed in.") in capsys.readouterr().out
    assert not any(method == "groups.succession.keep" for method, _ in calls)
    assert "Go back to Mac mini?" in group.keep_question("room", status("ok", targets=()), HOST)


def test_status_offers_moving_back_to_the_computer_a_bot_runs_on(gateway, capsys):
    calls, replies = gateway
    bots = [{"member_id": "writer", "name": "Writer", "on": {"install_id": OTHER, "name": "Laptop", "reachable": True}},
            {"member_id": "reviewer", "name": "Reviewer",
             "on": {"install_id": OTHER, "name": "Laptop", "reachable": True}}]
    hosted = status("ok", targets=(), unavailable_bots=bots)
    hosted["actions"] = [{"action": "move", "targets": [OTHER]}]
    replies["groups.succession.status"] = hosted
    assert group.cmd_status(argparse.Namespace(group="room")) == 0
    assert ("Writer and Reviewer can take part again if the group moves back to Laptop. To move it back: "
            "`hermes groups move <group> \"Laptop\"`.") in capsys.readouterr().out
    hosted["actions"], bots[0]["on"]["reachable"] = [], False
    assert group.cmd_status(argparse.Namespace(group="room")) == 0
    assert "Writer and Reviewer: unavailable until the group moves back to Laptop." in capsys.readouterr().out


def test_status_while_automatic_moves_settle_and_while_a_takeover_is_pending(gateway, capsys):
    calls, replies = gateway
    replies["groups.succession.status"] = status("ok", targets=(), automatic=automatic(enabled=True, pending=False))
    assert group.cmd_status(argparse.Namespace(group="room")) == 0
    assert "Turning off… (waiting for the other computers)" in capsys.readouterr().out
    replies["groups.succession.status"] = status(targets=(), unavailable_reason="takeover_waiting")
    assert group.cmd_status(argparse.Namespace(group="room")) == 0
    assert ("Mac mini went offline. The other computers are deciding which one takes over; this can take a few "
            "minutes.") in capsys.readouterr().out


def test_a_move_waits_for_replies_in_progress_unless_told_to_move_now(gateway, monkeypatch, capsys):
    calls, replies = gateway
    waiting = status("moving", targets=(), moving={"to": {"install_id": HERE, "name": "Home VPS"},
                                                   "step": "waiting_for_turns", "running": 2, "reason": "handover"})
    waiting["actions"] = [{"action": "move_now"}]
    replies["groups.succession.status"] = status("ok", targets=(), automatic=automatic())
    replies["groups.succession.move"] = waiting
    assert group.cmd_move(argparse.Namespace(group="room", computer="Home VPS", yes=True, now=False)) == 0
    out = capsys.readouterr().out
    assert "Moving to Home VPS after the replies in progress finish (2)." in out
    assert "To move now: `hermes groups move <group> \"Home VPS\" --now`." in out
    replies["groups.succession.status"] = waiting
    replies["groups.succession.move_now"] = status("moved_away", targets=(), moved={
        "to": {"install_id": HERE, "name": "Home VPS"}, "at": 1, "separate_events": 0, "branch_id": None})
    monkeypatch.setattr("builtins.input", lambda prompt: (print(prompt), "y")[1])
    assert group.cmd_move(argparse.Namespace(group="room", computer="Home VPS", yes=False, now=True)) == 0
    out = capsys.readouterr().out
    assert "Move now? Replies still in progress will show as unknown on Home VPS and won't rerun by themselves." in out
    assert ("groups.succession.move_now", {"room_id": "room"}) in calls
    assert sum(method == "groups.succession.move" for method, _ in calls) == 1


def test_moving_calls_wait_longer_than_ordinary_ones(monkeypatch):
    seen = []

    class Client:
        async def rpc(self, method, _timeout=30, **params):
            seen.append((method, _timeout))
            return {}

    class Connect:
        async def __aenter__(self):
            return Client()

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("hermes_cli.gateway_client.connect_gateway", lambda: Connect())
    group._call("groups.succession.promote", room_id="room")
    group._call("groups.succession.status", room_id="room")
    assert seen == [("groups.succession.promote", 300.0), ("groups.succession.status", 30.0)]


def test_status_tells_a_computer_that_needs_reconnecting_from_one_that_is_offline(gateway, capsys):
    calls, replies = gateway
    current = status("ok", targets=(), automatic=automatic("not_ready", reason="voters_offline", offline=[
        {"install_id": HERE, "name": "Home VPS"}, {"install_id": OTHER, "name": "Laptop"}]))
    current["backups"][0]["readiness"] = "needs_reauthorization"
    current["backups"][1]["readiness"] = "offline"
    replies["groups.succession.status"] = current
    assert group.cmd_status(argparse.Namespace(group="room")) == 0
    out = capsys.readouterr().out
    assert "Home VPS: needs to be reconnected: its permission to keep a copy ran out" in out
    assert "Right now Laptop is offline and Home VPS needs to be reconnected." in out
