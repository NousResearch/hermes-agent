"""Invariants for "is the agent making people wait?": every human message that reaches a busy chat
is counted exactly once, when its fate is known, and the day row reports the most chats a profile
had running at once."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_ingress_burst import discard_busy_burst, dispatch_busy_burst
from hermes_cli.observability import relay_shared_metrics as rsm
from hermes_cli.observability import shared_metrics_busy as busy
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_engagement as engagement
from hermes_cli.observability import shared_metrics_gateway as smg
from hermes_cli.observability.shared_metrics import SharedMetricsStore

KEY = "agent:main:telegram:dm:42"
RESOURCE = {"architecture": "x86_64", "hermes_version": "0.0.0", "install_method": "git", "os_family": "linux"}
DAY1 = datetime(2026, 9, 27, 9, 0, tzinfo=UTC).timestamp()


@pytest.fixture
def rows(monkeypatch):
    got, collecting = [], {"on": True}
    monkeypatch.setattr(rsm, "enabled", lambda: collecting["on"])
    monkeypatch.setattr(rsm, "record_process_mark", lambda mark, data: got.append((mark, dict(data))))
    monkeypatch.setattr(busy, "_queued", type(busy._queued)())

    def read():
        smg.drain()
        out = [(contract._DECISION_MARK_METRICS[mark], data) for mark, data in got]
        got.clear()
        return out

    read.collecting = collecting
    return read


class _Adapter:
    """The busy-dispatch surface of a platform adapter; ``handler`` stands in for the runner."""

    platform = Platform.TELEGRAM
    name = "telegram"
    serial_receive_loop = False
    gateway_runner = None
    _busy_text_mode = "queue"
    _busy_text_debounce_seconds = 120.0
    _busy_text_hard_cap_seconds = 120.0
    _is_queue_text_debounce_candidate = BasePlatformAdapter._is_queue_text_debounce_candidate
    _can_merge_text_debounce_events = BasePlatformAdapter._can_merge_text_debounce_events
    _unwrap_ephemeral = BasePlatformAdapter._unwrap_ephemeral

    def __init__(self, handler):
        self._message_handler = handler
        self._background_tasks = set()

    async def _release_turn_marker(self, event):
        return None


def _message(text):
    return MessageEvent(text=text, source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="7"))


def test_every_busy_arrival_counts_once_when_its_fate_is_known(rows):
    """Through the adapter's real busy dispatch: a held burst that /stop drops counts its head as
    dropped and the message folded into it as merged (both never reached the agent); a burst that
    commits and later runs as its own turn counts queued + merged with the wait until that turn; a
    busy message answered before admission (an approval reply) and a command are not counted."""

    async def scenario():
        stop = _Adapter(handler=None)
        first, second = _message("first"), _message("second")
        held = [asyncio.create_task(dispatch_busy_burst(stop, e, KEY)) for e in (first, second)]
        await asyncio.sleep(0)
        discard_busy_burst(stop, KEY)  # /stop while the burst is held
        await asyncio.gather(*held)

        async def runner_queues(event):  # the runner admits it to the FIFO; the turn starts later
            busy.note_busy_admitting(event)
            busy.note_busy_admitted(event, "adm-1", "queued")
            await asyncio.sleep(0)
            busy.busy_turn_started("adm-1")

        queue = _Adapter(runner_queues)
        tasks = [asyncio.create_task(dispatch_busy_burst(queue, e, KEY)) for e in (_message("a"), _message("b"))]
        await asyncio.sleep(0)
        queue.__dict__["_shared_bursts"][KEY].timer.cancel()
        from gateway.session_ingress_burst import _flush
        await _flush(queue, KEY, queue.__dict__["_shared_bursts"][KEY])
        await asyncio.gather(*tasks)
        await asyncio.gather(*queue._background_tasks)

        async def approval_reply(event):
            return None  # answered by the approval intercept: never admitted

        answered = _Adapter(approval_reply)
        answered._busy_text_mode = "interrupt"  # not a debounced burst: straight to the runner
        await dispatch_busy_burst(answered, _message("yes"), KEY)
        await dispatch_busy_burst(answered, _message("/status"), KEY)

    asyncio.run(scenario())
    got = sorted((data["disposition"], data["wait_bucket"]) for _metric, data in rows())
    assert got == [("dropped", "never"), ("merged", "lt_5s"), ("merged", "never"), ("queued", "lt_5s")]


def test_busy_dispositions_follow_the_busy_resolution_and_collection_off_records_nothing(rows):
    steer, interrupt, cleared = _message("s"), _message("i"), _message("c")
    for event in (steer, interrupt, cleared):
        busy.stamp_busy_arrival(_Adapter(None), event)
    busy.note_busy_outcome(steer, SimpleNamespace(steered=False, redirected=True, effective_mode="interrupt"))
    busy.note_busy_outcome(interrupt, SimpleNamespace(steered=False, redirected=False, effective_mode="interrupt"))
    busy.note_busy_admitting(interrupt)
    busy.note_busy_admitted(interrupt, "adm-2", "queued")
    interrupt._busy_arrival.arrived -= 45  # the interrupted turn took 45 s to stop
    busy.busy_turn_started("adm-2")
    busy.note_busy_admitting(cleared)
    busy.note_busy_admitted(cleared, "adm-3", "queued")
    busy.settle_busy_arrival(cleared)  # /new cancelled the queued row: its handler returned, never ran
    for event in (steer, interrupt, cleared):
        busy.settle_busy_arrival(event)
    got = rows()
    assert sorted((d["disposition"], d["wait_bucket"]) for _m, d in got) == [
        ("dropped", "never"), ("interrupted", "30s_to_2m"), ("steered", "lt_5s")]
    for metric, data in got:
        assert metric == "hermes.gateway.busy_arrival.count"
        assert data["platform"] == "telegram"
        assert contract.counter_dimensions_are_valid(metric, data)

    rows.collecting["on"] = False
    late = _message("late")
    busy.stamp_busy_arrival(_Adapter(None), late)
    busy.note_busy_outcome(late, SimpleNamespace(steered=True, redirected=False, effective_mode="steer"))
    assert rows() == []


def _store(home):
    root = home / "telemetry" / "shared_metrics"
    return SharedMetricsStore(root / "metrics.sqlite3", root / "outbox")


def test_day_row_reports_each_profiles_peak_of_chats_running_at_once(tmp_path, monkeypatch):
    """Profile A runs two chats at once, then one; profile B one. Each profile's closed day row
    carries its own peak (claimed in the owning profile's scope, not the launch profile's), and a
    row recorded before the field existed still packages."""
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setattr("gateway.run._write_runtime_status_quiet", lambda **_: None)
    monkeypatch.setattr(busy, "_running", {})
    monkeypatch.setattr(busy, "_peaks", {})
    clock = {"now": DAY1}
    monkeypatch.setattr(busy, "_now", lambda: clock["now"])
    monkeypatch.setattr(engagement, "_now", lambda: clock["now"])
    a, b = tmp_path / "a", tmp_path / "b"
    running: list[str] = []
    runner = SimpleNamespace(_active_work_count=lambda: len(running),
                             _running_agent_items=lambda: [(key, object()) for key in running])

    def turn_boundary(home, keys):
        running[:] = keys
        token = set_hermes_home_override(str(home))
        try:
            GatewayAgentCacheMixin._persist_active_agents(runner)
        finally:
            reset_hermes_home_override(token)

    turn_boundary(a, ["a:1"])
    turn_boundary(a, ["a:1", "a:2"])
    turn_boundary(b, ["a:1", "a:2", "b:1"])
    turn_boundary(a, ["b:1"])  # A's turns end; the release runs in whatever scope
    for home in (a, b):
        engagement.record(_store(home), RESOURCE, surface="gateway", route=None)
    clock["now"] = DAY1 + 86_400
    for home in (a, b):
        engagement.record(_store(home), RESOURCE, surface="gateway", route=None)

    for home, peak in ((a, "2"), (b, "1")):
        [row] = [c["dimensions"] for c in _store(home).counter_snapshot()
                 if c["metric_name"] == contract.ENGAGEMENT_DAY_METRIC]
        assert row[contract.BUSY_PEAK_FIELD] == peak
        assert contract.counter_dimensions_are_valid(contract.ENGAGEMENT_DAY_METRIC, row)
        legacy = {k: v for k, v in row.items() if k != contract.BUSY_PEAK_FIELD}
        assert contract.counter_dimensions_are_valid(contract.ENGAGEMENT_DAY_METRIC, legacy)


class _PausingAuthority:
    """The admission surface ``admit_message`` reads: the row commits as ``queued``, then the FIFO
    pauses (a runtime stop releases every waiter with ``runtime_draining``, a worker on the chat
    with ``session_busy``) while the row stays queued for later."""

    def __init__(self):
        self.db, self.native_waiters, self.waiters = None, set(), {}
        self.sessions = {"sid": SimpleNamespace(pause_notified=False, source=None)}

    async def admit_native(self, event):
        return SimpleNamespace(status="queued", admission_id="adm-p", ref=SimpleNamespace(session_id="sid"))


@pytest.mark.parametrize("reason", ["runtime_draining", "session_busy"])
def test_a_busy_message_saved_behind_a_pause_is_not_dropped(rows, monkeypatch, reason):
    """The user is told the message is saved and its FIFO row stays ``queued``: no ``dropped``
    row. When this process later runs that row (a ``session_busy`` pause lifting), it counts as
    ``queued`` with its wait; after a restart the next process runs it and nothing is counted."""
    import gateway.session_authority  # binds the real reader before the patch below
    import hermes_state_runtime
    from gateway.session_ingress import admit_message
    from hermes_state_runtime import RuntimeStoreError

    monkeypatch.setattr(hermes_state_runtime, "get_session_admission",
                        lambda db, admission_id: {"status": "queued"})
    authority, replies = _PausingAuthority(), []

    async def runner_admits(event):
        busy.note_busy_admitting(event)
        replies.append(await admit_message(authority, event))

    async def scenario():
        adapter = _Adapter(runner_admits)
        adapter._busy_text_mode = "interrupt"  # not a debounced burst: straight to the runner
        await dispatch_busy_burst(adapter, _message("b"), KEY)
        await asyncio.sleep(0)
        authority.waiters["adm-p"].set_exception(RuntimeStoreError(reason))
        await asyncio.gather(*adapter._background_tasks)

    asyncio.run(scenario())
    assert "saved" in replies[0]
    assert rows() == []
    busy.busy_turn_started("adm-p")  # the pause lifted and this process ran the row
    assert [(d["disposition"], d["wait_bucket"]) for _m, d in rows()] == [("queued", "lt_5s")]


def test_an_auto_resumed_chat_counts_toward_its_owning_profiles_peak(tmp_path, monkeypatch):
    """The startup auto-resume loop claims chats from the launch profile's scope; a resumed chat
    of another profile counts toward that profile's peak, not the launch profile's."""
    from gateway.session import SessionEntry
    from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

    monkeypatch.setattr("gateway.run._write_runtime_status_quiet", lambda **_: None)
    monkeypatch.setattr(busy, "_running", {})
    monkeypatch.setattr(busy, "_peaks", {})
    monkeypatch.setattr(busy, "_now", lambda: DAY1)
    launch, beta = tmp_path / "launch", tmp_path / "beta"
    monkeypatch.setenv("HERMES_HOME", str(launch))
    runner, _adapter = make_restart_runner()
    runner.session_authorities = []
    runner._resolve_profile_home_for_source = lambda source: beta
    entry = SessionEntry(session_key="agent:beta:telegram:dm:c", session_id="s-beta",
                         created_at=datetime.now(), updated_at=datetime.now(),
                         origin=make_restart_source(chat_id="c"), platform=Platform.TELEGRAM, chat_type="dm",
                         resume_pending=True, resume_reason="restart_interrupted",
                         last_resume_marked_at=datetime.now())
    runner.session_store._entries = {entry.session_key: entry}

    async def resume():
        runner._run_startup_resume_event = lambda *a, **k: asyncio.sleep(0)
        assert runner._schedule_resume_pending_sessions() == 1
        await asyncio.sleep(0)

    asyncio.run(resume())
    day = busy._day(DAY1)
    assert (busy.day_peak(beta, day), busy.day_peak(launch, day)) == (1, 0)


def test_a_held_burst_dropped_by_stop_counts_under_its_own_profile_not_the_chats_last_one(rows, monkeypatch, tmp_path):
    """Two profiles serve one Discord channel through different bots; the other profile ran the
    chat's latest turn. A burst held for this profile's bot and dropped by /stop is recorded under
    the profile its routing identity names, so that profile's collection setting decides."""
    from gateway.session_identity import RoutingIdentity
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    homes = []
    monkeypatch.setattr(rsm, "record_process_mark", lambda mark, data: homes.append(str(get_hermes_home())))
    alpha, beta = tmp_path / "alpha", tmp_path / "beta"
    token = set_hermes_home_override(str(beta))
    try:
        smg.start_reply_clock(SessionSource(platform=Platform.DISCORD, chat_id="chan", user_id="9"))
    finally:
        reset_hermes_home_override(token)
    source = SessionSource(platform=Platform.DISCORD, chat_id="chan", user_id="7")
    source._identity = RoutingIdentity(transport_profile="alpha", runtime_profile="alpha",
                                       authorization_home=alpha, runtime_home=alpha)

    async def scenario():
        adapter = _Adapter(handler=None)
        adapter.platform = Platform.DISCORD
        held = asyncio.create_task(dispatch_busy_burst(adapter, MessageEvent(text="x", source=source), KEY))
        await asyncio.sleep(0)
        discard_busy_burst(adapter, KEY)
        await held

    asyncio.run(scenario())
    smg.drain()
    assert homes == [str(alpha)]


def test_a_closing_day_keeps_a_peak_reached_just_before_midnight(tmp_path, monkeypatch):
    """A third chat starts at 23:59:30; its first engagement event lands after midnight, and a turn
    boundary after midnight already started the new day. The closed day still reports 3."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setattr(busy, "_running", {})
    monkeypatch.setattr(busy, "_peaks", {})
    late = datetime(2026, 9, 27, 23, 59, 0, tzinfo=UTC).timestamp()
    clock = {"now": late}
    monkeypatch.setattr(busy, "_now", lambda: clock["now"])
    monkeypatch.setattr(engagement, "_now", lambda: clock["now"])
    home = tmp_path / "a"

    def boundary(keys):
        token = set_hermes_home_override(str(home))
        try:
            busy.observe_running_chats(keys)
        finally:
            reset_hermes_home_override(token)

    boundary(["c1", "c2"])
    engagement.record(_store(home), RESOURCE, surface="gateway", route=None)
    clock["now"] = late + 30
    boundary(["c1", "c2", "c3"])
    clock["now"] = late + 90
    boundary(["c3"])
    engagement.record(_store(home), RESOURCE, surface="gateway", route=None)
    [row] = [c["dimensions"] for c in _store(home).counter_snapshot()
             if c["metric_name"] == contract.ENGAGEMENT_DAY_METRIC]
    assert row[contract.BUSY_PEAK_FIELD] == "3_to_5"


async def _real_runner(monkeypatch):
    """Real GatewayRunner busy dispatch, real DiscordAdapter, real authority over a real SessionDB;
    the drain is not scheduled, so a queued row waits until the test runs or cancels it."""
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    sent = []

    async def send(chat_id, content, **_kwargs):
        sent.append(content)
        return SendResult(success=True, message_id=str(len(sent)))
    monkeypatch.setattr(adapter, "send", send)  # no network: notices and acks land here
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner")
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    key = runner.session_store._generate_session_key(source)

    class RunningAgent:
        def interrupt(self, message=None):
            pass

    runner._session_state(key).turn.agent = RunningAgent()  # an earlier queued turn is running
    return runner, authority, source, key


def _busy_rows(rows):
    return [(d["disposition"], d["wait_bucket"]) for metric, d in rows() if metric == busy.BUSY_ARRIVAL_METRIC]


@pytest.mark.asyncio
async def test_a_message_the_adapter_saw_idle_counts_when_it_waits_behind_a_running_fifo_turn(rows, monkeypatch):
    """A's adapter task ended while queued B runs as a FIFO turn (outside the adapter's session
    guard): C reaches the runner as a fresh message, is admitted behind B, and counts once when
    its own turn starts."""
    from gateway.session_contract import SessionRef
    from gateway.session_authority import SessionAuthority
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home

    runner, authority, source, key = await _real_runner(monkeypatch)
    monkeypatch.setattr(runner, "_effective_busy_input_mode", lambda source: "queue")
    late = MessageEvent(text="C", source=source, message_id="C")
    with native_callback(runner, late, get_hermes_home()):
        waiting = asyncio.create_task(runner._handle_message(late))
        for _ in range(200):
            if authority.waiters:
                break
            await asyncio.sleep(0.01)
    assert authority.waiters and _busy_rows(rows) == []  # still waiting behind B: not yet known
    runner._session_state(key).turn.agent = None  # B finished; the drain runs C

    async def handle(event):
        return "ran"
    monkeypatch.setattr(runner, "_handle_message", handle)
    owner = runner.session_store.peek_session_id(key)
    await SessionAuthority._drain(authority, SessionRef(authority.profile_id, owner))
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    assert _busy_rows(rows) == [("queued", "lt_5s")]


@pytest.mark.asyncio
async def test_a_busy_message_cancelled_before_admission_reads_its_row_counts_as_dropped(rows, monkeypatch):
    """/stop cancels the queued row while ``admit_message`` is still committing it: the re-read
    finds it terminal ``cancelled``. It never reached the agent: ``dropped`` / ``never``."""
    from gateway.session_ingress import admit_message
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home

    runner, authority, source, _key = await _real_runner(monkeypatch)
    follow_up = MessageEvent(text="B", source=source, message_id="B")
    busy.stamp_busy_arrival(runner.adapters[Platform.DISCORD], follow_up)
    busy.note_busy_admitting(follow_up)
    commit = authority.admit_native

    async def commit_then_stop(event):
        receipt = await commit(event)
        stop = MessageEvent(text="/stop", source=source)
        with native_callback(runner, stop, get_hermes_home()):
            await runner._handle_message(stop)
        return receipt

    monkeypatch.setattr(authority, "admit_native", commit_then_stop)
    with native_callback(runner, follow_up, get_hermes_home()):
        assert await admit_message(authority, follow_up) is None
    busy.settle_busy_arrival(follow_up)
    assert _busy_rows(rows) == [("dropped", "never")]
