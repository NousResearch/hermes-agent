"""The status a computer reports for its group, the host's offline window, planned restarts and the
owner's one notice per incident."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import hosted_room_succession as succession
from gateway import hosted_room_succession_move as move
from gateway import hosted_room_succession_status as status_module
from tests.gateway.fixtures.succession import ROOM, context, copy_to, home_room, make_gateways


@pytest.fixture
def gateways(tmp_path):
    values = make_gateways(tmp_path, "h", "s", "p")
    home_room(values, "h", messages=2, successors=("s",))
    copy_to(values["h"], values["s"])
    copy_to(values["h"], values["p"])
    yield values
    for gateway in values.values():
        gateway.close()


def beat(gateway, gateways, *, now, down=("h",)):
    with gateway.acting():
        return status_module.heartbeat(context(gateway, gateways, down=down), ROOM, now=now, survey_peers=True)


def status_of(gateway, gateways, **options):
    with gateway.acting():
        return status_module.status(context(gateway, gateways, **options), ROOM)


def test_a_reachable_host_shows_no_banner_and_names_every_backup(gateways):
    h, s, p = gateways["h"], gateways["s"], gateways["p"]
    beat(s, gateways, now=1000.0, down=())
    current = status_of(s, gateways)
    assert current["state"] == "ok" and current["unavailable_reason"] == "host_reachable"
    assert current["host"] == {"install_id": h.install_id, "name": "H", "reachable": True, "since": None}
    assert current["this_install"] == {"install_id": s.install_id, "name": "S", "role": "backup"}
    assert current["owner"] == {"name": "Dana"}
    rows = {row["install_id"]: row for row in current["backups"]}
    assert set(rows) == {s.install_id, p.install_id}
    assert rows[s.install_id]["successor"] and rows[s.install_id]["readiness"] == "caught_up"
    assert rows[p.install_id]["kind"] == "member" and rows[p.install_id]["successor"] is False
    assert rows[s.install_id]["operator_name"] == "S Operator"
    hosted = status_of(h, gateways)
    assert hosted["this_install"]["role"] == "host" and hosted["state"] == "ok"


def test_the_host_counts_as_offline_only_after_the_window_and_the_owner_is_offered_the_best_computer(gateways):
    s, p = gateways["s"], gateways["p"]
    beat(s, gateways, now=1000.0, down=())
    beat(s, gateways, now=1060.0)
    assert status_of(s, gateways)["state"] == "ok"  # one missed answer is not an outage
    beat(s, gateways, now=1000.0 + status_module.UNREACHABLE_AFTER_SECONDS + 1)
    current = status_of(s, gateways)
    assert current["state"] == "host_unreachable"
    assert current["host"]["reachable"] is False and current["host"]["since"] == 1000.0
    assert current["actions"] == [{"action": "continue", "targets": [s.install_id]}]
    assert status_of(s, gateways, subject="uid:999")["unavailable_reason"] == "not_owner"
    beat(p, gateways, now=1000.0, down=())
    beat(p, gateways, now=1000.0 + status_module.UNREACHABLE_AFTER_SECONDS + 1)
    assert status_of(p, gateways, operator=True)["actions"] == [{"action": "continue", "targets": [s.install_id]}]


def test_a_planned_restart_is_not_loss(gateways):
    h, s = gateways["h"], gateways["s"]
    with h.acting():
        move.append_state(h.db, ROOM, "host_restarting", until=5000.0)
    copy_to(h, s)
    beat(s, gateways, now=1000.0, down=())
    beat(s, gateways, now=2000.0)
    with s.acting(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(status_module.time, "time", lambda: 2000.0)
        current = status_module.status(context(s, gateways, down=("h",)), ROOM)
        assert current["state"] == "host_restarting" and current["host"]["restarting_until"] == 5000.0
        assert current["actions"] == [] and status_module.notice_due(context(s, gateways), ROOM, now=2000.0) is None
        with pytest.raises(succession.SuccessionError) as refused:
            move.preview(context(s, gateways, down=()), ROOM, s.install_id)
        assert refused.value.reason == "host_reachable"


def test_an_unknown_group_is_room_not_found(gateways):
    s = gateways["s"]
    with s.acting(), pytest.raises(succession.SuccessionError) as missing:
        status_module.status(context(s, gateways), "other-room")
    assert missing.value.reason == "room_not_found"


class _Adapter:
    platform = "telegram"

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SimpleNamespace(success=True)


@pytest.fixture
def loop():
    running = asyncio.new_event_loop()
    thread = threading.Thread(target=running.run_forever, daemon=True)
    thread.start()
    yield running
    running.call_soon_threadsafe(running.stop)
    thread.join(5)
    running.close()


def test_the_best_placed_computer_tells_the_owner_once_per_incident(gateways, loop):
    s = gateways["s"]
    adapter = _Adapter()

    async def refs(room_id):
        return [(adapter, "chat-1", {"thread_id": "7"}, 3)] if room_id == ROOM else []

    runner = SimpleNamespace(_group_chat_continue_refs=refs, _typed_command_prefix_for=lambda platform: "/")
    beat(s, gateways, now=1000.0, down=())
    offline = 1000.0 + status_module.UNREACHABLE_AFTER_SECONDS + 60
    beat(s, gateways, now=offline)
    with s.acting(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(status_module.time, "time", lambda: offline)
        assert status_module.notify(context(s, gateways, down=("h",)), ROOM, runner, loop, now=offline)
        assert not status_module.notify(context(s, gateways, down=("h",)), ROOM, runner, loop, now=offline + 60)
    assert adapter.sent == [("chat-1", "“Room” is paused: H has been offline for 6 min. Reply /group 3 continue to "
                                       "continue it on S.", {"thread_id": "7"})]


def test_without_messaging_the_notice_goes_to_the_home_channel(gateways, loop):
    s = gateways["s"]
    sent = []

    async def send_home(platform, home, transport, message, failure_fmt):
        sent.append(message)
        return True

    runner = SimpleNamespace(_home_channel_transports=lambda: [("telegram", None, "home", "transport")],
                             _send_home_channel_message=send_home)
    beat(s, gateways, now=1000.0, down=())
    offline = 1000.0 + status_module.UNREACHABLE_AFTER_SECONDS + 1
    beat(s, gateways, now=offline)
    with s.acting(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(status_module.time, "time", lambda: offline)
        assert status_module.notify(context(s, gateways, down=("h",)), ROOM, runner, loop, now=offline)
    assert sent == ["“Room” is paused: H has been offline for 5 min. Open Hermes Desktop on a computer you own, "
                    "or run `hermes groups continue Room` on S."]


def test_a_computer_that_is_not_best_placed_waits_and_stays_quiet_while_a_better_one_is_up(tmp_path, loop):
    gateways = make_gateways(tmp_path, "h", "s", "t")
    try:
        home_room(gateways, "h", successors=("s", "t"))
        copy_to(gateways["h"], gateways["s"])
        copy_to(gateways["h"], gateways["t"], through=2)
        t = gateways["t"]
        sent = []

        async def send_home(platform, home, transport, message, failure_fmt):
            sent.append(message)
            return True

        runner = SimpleNamespace(_home_channel_transports=lambda: [("telegram", None, "home", "transport")],
                                 _send_home_channel_message=send_home)
        beat(t, gateways, now=1000.0, down=())
        offline = 1000.0 + status_module.UNREACHABLE_AFTER_SECONDS + 1
        beat(t, gateways, now=offline)
        with t.acting(), pytest.MonkeyPatch.context() as patch:
            patch.setattr(status_module.time, "time", lambda: offline)
            ctx = context(t, gateways, down=("h",))
            assert status_module.status(ctx, ROOM)["actions"][0]["targets"][0] == gateways["s"].install_id
            assert not status_module.notify(ctx, ROOM, runner, loop, now=offline)
            later = offline + status_module.NOTICE_DELAY_SECONDS + 1
            patch.setattr(status_module.time, "time", lambda: later)
            assert not status_module.notify(ctx, ROOM, runner, loop, now=later)  # s still answers
        beat(t, gateways, now=later + 60, down=("h", "s"))
        with t.acting(), pytest.MonkeyPatch.context() as patch:
            patch.setattr(status_module.time, "time", lambda: later + 60)
            assert status_module.notify(context(t, gateways, down=("h", "s")), ROOM, runner, loop, now=later + 60)
        assert len(sent) == 1 and sent[0].endswith("on T.")
    finally:
        for gateway in gateways.values():
            gateway.close()


def test_messaging_renders_the_notice_with_its_own_button_when_it_can(gateways, loop):
    s = gateways["s"]
    rendered = []

    async def notify(room_id, kind, facts):
        rendered.append((room_id, kind, facts))
        return 1

    runner = SimpleNamespace(_group_chat_notify=notify, _group_chat_continue_refs=None)
    beat(s, gateways, now=1000.0, down=())
    offline = 1000.0 + status_module.UNREACHABLE_AFTER_SECONDS + 1
    beat(s, gateways, now=offline)
    with s.acting(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(status_module.time, "time", lambda: offline)
        assert status_module.notify(context(s, gateways, down=("h",)), ROOM, runner, loop, now=offline)
        assert not status_module.notify(context(s, gateways, down=("h",)), ROOM, runner, loop, now=offline + 30)
    assert rendered == [(ROOM, "host_offline", {"host": "H", "minutes": 5})]


def test_a_copy_refused_for_lack_of_permission_needs_reauthorization_not_offline(gateways):
    """The host's copy to a computer was refused (its grant lapsed while the host couldn't reach it): the
    row says so, distinct from offline, and that computer can't take a move until it renews."""
    import time
    from gateway import hosted_room_replication as replication
    from gateway import hosted_rooms as rooms
    h, s = gateways["h"], gateways["s"]
    with h.acting(), rooms._transaction(h.db, immediate=True) as conn:
        replication._initialize(conn)
        conn.execute(f"""INSERT INTO {replication.TARGETS_TABLE} (room_id, target_install_id, lineage,
            selected_member_id, status, updated_at) VALUES (?,?,?,?,?,?)""",
                     (ROOM, s.install_id, "lineage", "custody@" + s.install_id, "needs_reauthorization", time.time()))
    rows = {row["install_id"]: row for row in status_of(h, gateways)["backups"]}
    assert rows[s.install_id]["readiness"] == "needs_reauthorization"
    assert rows[gateways["p"].install_id]["readiness"] != "needs_reauthorization"
    move_action = next((a for a in status_of(h, gateways)["actions"] if a["action"] == "move"), None)
    assert move_action is None or s.install_id not in move_action["targets"]
