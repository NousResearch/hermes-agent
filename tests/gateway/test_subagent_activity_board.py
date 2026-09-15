"""Subagent activity board on Telegram."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.relay.descriptor import CapabilityDescriptor
from gateway.subagent_activity_board import SubagentActivityBoard, _format_elapsed
from gateway.platforms.base import BasePlatformAdapter
from gateway.display_config import resolve_display_setting
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


class _EditingAdapter:
    """Records one send per message id and every edit against it."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []           # (message_id, text)
        self.edits: list[tuple[str, str]] = []          # (message_id, text)
        self.send_routes: list[tuple[str, dict, str]] = []
        self.edit_routes: list[tuple[str, dict, str]] = []
        self.fail_edits = 0                             # next N edits fail (retryable)
        self.retry_after = None

    async def send(self, chat_id, content, *, metadata=None):
        message_id = f"msg-{len(self.sent) + 1}"
        self.sent.append((message_id, content))
        self.send_routes.append((str(chat_id), dict(metadata or {}), message_id))
        return SimpleNamespace(success=True, message_id=message_id)

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        del finalize
        assert metadata is not None and metadata.get("_interim_send") is True
        if self.fail_edits:
            self.fail_edits -= 1
            return SimpleNamespace(success=False, message_id=None, retryable=True, retry_after=self.retry_after)
        self.edits.append((message_id, content))
        self.edit_routes.append((str(chat_id), dict(metadata), str(message_id)))
        return SimpleNamespace(success=True, message_id=message_id)

    @property
    def texts(self) -> list[str]:
        return [text for _id, text in self.sent + self.edits]


class _SendOnlyAdapter:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, _chat_id, content, *, metadata=None):
        del metadata
        self.sent.append(content)
        return SimpleNamespace(success=True, message_id="message-1")


class _FakeTime:
    """Manual clock; sleep() advances it and yields once so other tasks run."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


class _BlockingTime(_FakeTime):
    """Manual clock whose first sleep can hold a publisher while events coalesce."""

    def __init__(self) -> None:
        super().__init__()
        self.sleeping = asyncio.Event()
        self.release = asyncio.Event()

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.sleeping.set()
        await self.release.wait()
        self.now += seconds


@pytest.fixture
def scheduled(monkeypatch):
    """Route TurnRunner._schedule onto the test loop and collect the tasks."""
    from gateway import run as run_mod

    tasks: list[asyncio.Task] = []

    def _schedule(coro, _loop, **_kwargs):
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    monkeypatch.setattr(run_mod, "safe_schedule_threadsafe", _schedule)
    return tasks


async def _drain(tasks: list[asyncio.Task]) -> None:
    while tasks:
        pending = [task for task in tasks if not task.done()]
        if not pending:
            return
        await asyncio.gather(*pending)


def _turn(adapter, *, platform=Platform.TELEGRAM, runner=None, fake_time=None, thread_id="77", **fields):
    ctx = TurnContext(
        source=SessionSource(platform=platform, chat_id="chat"),
        _run_still_current=lambda: True,
        _status_adapter=adapter,
        _status_chat_id="chat",
        _status_thread_metadata={"thread_id": str(thread_id)},
        _loop_for_step=asyncio.get_running_loop(),
        **fields,
    )
    if fake_time is not None:
        ctx._subagent_activity_board = SubagentActivityBoard(
            adapter, "chat", ctx._status_thread_metadata, clock=fake_time.clock, sleep=fake_time.sleep,
        )
    else:  # no pacing: tests that only care about content
        ctx._subagent_activity_board = SubagentActivityBoard(
            adapter, "chat", ctx._status_thread_metadata, min_edit_interval=0.0, sleep=asyncio.sleep,
        )
    return TurnRunner(runner if runner is not None else SimpleNamespace(), ctx)


def _child(index: int, count: int = 1, **extra) -> dict:
    return {"subagent_id": f"sa-{index}", "task_index": index, "task_count": count, "depth": 0, **extra}


@pytest.mark.asyncio
async def test_detached_child_lifecycle_edits_one_bubble_after_generation_advances(scheduled):
    """Late child events keep editing only their initiating turn's bubble, payload-free."""
    from tools.delegate_tool_progress import _build_child_progress_callback

    adapter = _EditingAdapter()
    fake = _FakeTime()
    current = [True]
    turn = _turn(adapter, fake_time=fake)
    turn._ctx._run_still_current = lambda: current[0]
    parent = SimpleNamespace(
        _delegate_spinner=None,
        tool_progress_callback=turn.progress_callback,
        session_id="PRIVATE-session",
    )
    relay = _build_child_progress_callback(
        0, "PRIVATE GOAL", parent, task_count=1, subagent_id="sa-real", depth=0,
        model="PRIVATE-model", session_ref={"delegation_id": "deleg-real"},
    )

    relay("subagent.start")
    await _drain(scheduled)
    current[0] = False  # the detached child outlives its initiating foreground generation
    relay("tool.started", "read_file", "/PRIVATE/path", {"path": "/PRIVATE/path"})
    await _drain(scheduled)
    relay("subagent.complete", preview="PRIVATE output", status="ok", summary="PRIVATE summary")
    await _drain(scheduled)

    assert len(adapter.sent) == 1
    assert [message_id for message_id, _ in adapter.edits] == ["msg-1", "msg-1"]
    assert "0/1 done" in adapter.texts[0] and "🚀 spawned" in adapter.texts[0]
    assert "⚙️ working · <30s · 1 tool" in adapter.texts[1]
    assert "1/1 done" in adapter.texts[2] and "✅ done" in adapter.texts[2]
    assert "PRIVATE" not in " ".join(adapter.texts)


@pytest.mark.asyncio
async def test_failure_renders_structurally_and_the_notice_rail_is_untouched(scheduled):
    notices: list[str] = []

    class _Runner:
        async def _deliver_platform_notice(self, _source, content):
            notices.append(content)

    adapter = _EditingAdapter()
    _turn(adapter, runner=_Runner()).progress_callback(
        "subagent.complete", preview="PRIVATE traceback", status="failed", goal="goal text",
        summary="PRIVATE credential", duration_seconds=3.0, **_child(0),
    )
    await _drain(scheduled)

    assert len(adapter.sent) == 1
    assert "1/1 done" in adapter.texts[0] and "❌ failed" in adapter.texts[0]
    assert "PRIVATE" not in adapter.texts[0]
    # The existing one-shot failure notice still fires exactly as before.
    assert len(notices) == 1 and "goal text" in notices[0]


@pytest.mark.parametrize("status, phase", [("error", "❌ failed"), ("unknown", "❌ failed"),
                                           ("brand-new-status", "❌ failed"), ("timeout", "⏱ timeout"),
                                           ("interrupted", "⛔ stopped")])
def test_every_completion_status_is_terminal(status, phase):
    board = SubagentActivityBoard(None, "chat", None)
    assert board.observe("subagent.complete", {**_child(0), "status": status})
    text = board._render()
    assert phase in text and "1/1 done" in text
    # A settled child never advances again, so a late tool event owes no edit.
    assert board.observe("subagent.tool", {**_child(0), "tool_count": 1}) is False


@pytest.mark.asyncio
async def test_no_bubble_without_edit_support_off_telegram_or_when_disabled(scheduled):
    class _InheritedBaseEdit(_SendOnlyAdapter):
        edit_message = BasePlatformAdapter.edit_message

    class _RelayWithoutEdit(_EditingAdapter):
        def _descriptor_for_chat(self, _chat_id):
            return SimpleNamespace(supports_edit=False)

    class _RelayWithoutEditOp(_EditingAdapter):
        def _descriptor_for_chat(self, _chat_id):
            return CapabilityDescriptor(
                contract_version=1,
                platform="telegram",
                label="Telegram",
                max_message_length=4096,
                supports_draft_streaming=False,
                supports_edit=True,
                supports_threads=True,
                markdown_dialect="telegram_html",
                len_unit="chars",
                supported_ops=("send", "typing"),
            )

    send_only = _SendOnlyAdapter()
    inherited = _InheritedBaseEdit()
    relay_without_edit = _RelayWithoutEdit()
    relay_without_edit_op = _RelayWithoutEditOp()
    discord, disabled = _EditingAdapter(), _EditingAdapter()
    for turn in (
        _turn(send_only),
        _turn(inherited),
        _turn(relay_without_edit),
        _turn(relay_without_edit_op),
        _turn(discord, platform=Platform.DISCORD),
        _turn(disabled, subagent_activity_board_enabled=False),
    ):
        turn.progress_callback("subagent.start", **_child(0))
        turn.progress_callback("subagent.tool", "read_file", tool_count=1, **_child(0))
    await _drain(scheduled)

    assert send_only.sent == [] and inherited.sent == []
    assert relay_without_edit.sent == [] and relay_without_edit_op.sent == []
    assert discord.sent == [] and disabled.sent == []


@pytest.mark.asyncio
async def test_board_is_created_lazily_and_only_for_eligible_turns(scheduled):
    telegram = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat"), _run_still_current=lambda: True,
        _status_adapter=_EditingAdapter(), _status_chat_id="chat", _loop_for_step=asyncio.get_running_loop(),
    )
    discord = TurnContext(
        source=SessionSource(platform=Platform.DISCORD, chat_id="chat"), _run_still_current=lambda: True,
        _status_adapter=_EditingAdapter(), _status_chat_id="chat", _loop_for_step=asyncio.get_running_loop(),
    )
    assert telegram._subagent_activity_board is None and discord._subagent_activity_board is None
    TurnRunner(SimpleNamespace(), telegram).progress_callback("tool.started", "read_file")
    assert telegram._subagent_activity_board is None
    TurnRunner(SimpleNamespace(), telegram).progress_callback("subagent.start", **_child(0))
    TurnRunner(SimpleNamespace(), discord).progress_callback("subagent.start", **_child(0))
    await _drain(scheduled)

    assert isinstance(telegram._subagent_activity_board, SubagentActivityBoard)
    assert discord._subagent_activity_board is None
    assert len(telegram._status_adapter.sent) == 1 and discord._status_adapter.sent == []


@pytest.mark.asyncio
async def test_concurrent_first_events_create_exactly_one_board_and_bubble(monkeypatch):
    """A cold first wave arrives from N workers; lazy creation must be one compare-and-set."""
    from gateway import run as run_mod
    from gateway import subagent_activity_board as activity_mod

    loop = asyncio.get_running_loop()
    scheduled_futures = []
    futures_lock = threading.Lock()
    boards_created = 0
    boards_lock = threading.Lock()
    real_board = SubagentActivityBoard

    class _SlowBoard(real_board):
        def __init__(self, *args, **kwargs):
            nonlocal boards_created
            time.sleep(0.02)  # widen the check/create race window
            with boards_lock:
                boards_created += 1
            super().__init__(*args, min_edit_interval=0.0, **kwargs)

    def _schedule(coro, target_loop, **_kwargs):
        future = asyncio.run_coroutine_threadsafe(coro, target_loop)
        with futures_lock:
            scheduled_futures.append(future)
        return future

    monkeypatch.setattr(activity_mod, "SubagentActivityBoard", _SlowBoard)
    monkeypatch.setattr(run_mod, "safe_schedule_threadsafe", _schedule)
    adapter = _EditingAdapter()
    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat"),
        _run_still_current=lambda: True, _status_adapter=adapter, _status_chat_id="chat",
        _loop_for_step=loop,
    )
    loop_thread = threading.get_ident()
    retained = []

    def _retain(task):
        assert threading.get_ident() == loop_thread
        assert task is asyncio.current_task()
        retained.append(task)

    runner = SimpleNamespace(_running=True, _retain_background_task=_retain)
    turn = TurnRunner(runner, ctx)
    barrier = threading.Barrier(8)

    def _start(index):
        barrier.wait()
        turn.progress_callback("subagent.start", **_child(index, 8))

    threads = [threading.Thread(target=_start, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    await asyncio.to_thread(lambda: [thread.join() for thread in threads])
    with futures_lock:
        futures = list(scheduled_futures)
    await asyncio.gather(*(asyncio.wrap_future(future) for future in futures))

    assert boards_created == 1
    assert len(retained) == len(futures) >= 1
    assert len(adapter.sent) == 1
    assert "0/8 done" in adapter.texts[-1]
    assert adapter.texts[-1].count("🚀 spawned") == 8

    # Admission and retention stay on the gateway loop; once shutdown starts, a new board
    # may observe state but must not publish it.
    runner._running = False
    shutdown_ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat"),
        _run_still_current=lambda: True, _status_adapter=adapter, _status_chat_id="chat",
        _loop_for_step=loop,
    )
    before = len(scheduled_futures)
    TurnRunner(runner, shutdown_ctx).progress_callback("subagent.start", **_child(0))
    with futures_lock:
        shutdown_futures = list(scheduled_futures[before:])
    await asyncio.gather(*(asyncio.wrap_future(future) for future in shutdown_futures))
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_parent_turns_in_one_chat_edit_their_own_messages(scheduled):
    adapter = _EditingAdapter()
    turns = [
        _turn(adapter, run_generation=generation, thread_id=thread_id)
        for generation, thread_id in ((1, "77"), (2, "88"))
    ]
    for turn in turns:
        turn.progress_callback("subagent.start", **_child(0))
    await _drain(scheduled)
    for turn in reversed(turns):
        turn.progress_callback("subagent.complete", status="ok", **_child(0))
    await _drain(scheduled)

    assert [message_id for message_id, _ in adapter.sent] == ["msg-1", "msg-2"]
    assert [message_id for message_id, _ in adapter.edits] == ["msg-2", "msg-1"]
    assert adapter.send_routes == [
        ("chat", {"thread_id": "77", "_interim_send": True}, "msg-1"),
        ("chat", {"thread_id": "88", "_interim_send": True}, "msg-2"),
    ]
    assert adapter.edit_routes == [
        ("chat", {"thread_id": "88", "_interim_send": True}, "msg-2"),
        ("chat", {"thread_id": "77", "_interim_send": True}, "msg-1"),
    ]


@pytest.mark.asyncio
async def test_bursts_coalesce_to_one_edit_per_interval_and_finish_on_the_latest_state(scheduled):
    fake = _BlockingTime()
    adapter = _EditingAdapter()
    turn = _turn(adapter, fake_time=fake)

    turn.progress_callback("subagent.start", **_child(0, 10))
    await _drain(scheduled)
    turn.progress_callback("subagent.tool", "bash", tool_count=1, **_child(0, 10))
    await fake.sleeping.wait()
    for index in range(1, 10):
        turn.progress_callback("subagent.start", **_child(index, 10))
    for count in range(1, 31):
        for index in range(10):
            turn.progress_callback("subagent.tool", "bash", tool_count=count, **_child(index, 10))
    for index in range(10):
        turn.progress_callback("subagent.complete", status="ok", **_child(index, 10))
    fake.release.set()
    await _drain(scheduled)

    # Hundreds of revisions arriving during the edit floor collapse into one latest-state edit.
    assert len(adapter.sent) == 1
    assert len(adapter.edits) == 1
    assert fake.slept == [3.0]
    assert "10/10 done" in adapter.texts[-1]
    assert adapter.texts[-1].count("✅ done · 0s · 30 tools") == 8 and "…and 2 more" in adapter.texts[-1]


@pytest.mark.asyncio
async def test_failed_edit_retries_the_latest_state_and_never_sends_a_second_message(scheduled):
    fake = _FakeTime()
    adapter = _EditingAdapter()
    turn = _turn(adapter, fake_time=fake)

    turn.progress_callback("subagent.start", **_child(0))
    await _drain(scheduled)
    adapter.fail_edits, adapter.retry_after = 1, 7.5
    turn.progress_callback("subagent.tool", "bash", tool_count=1, **_child(0))
    await _drain(scheduled)

    assert len(adapter.sent) == 1
    assert [message_id for message_id, _ in adapter.edits] == ["msg-1"]
    assert 7.5 in fake.slept  # the server's retry_after is honoured before the retry
    assert "working · <30s · 1 tool" in adapter.texts[-1]

    # Explicitly retryable failures do not consume a terminal revision that has no later wake-up.
    adapter.fail_edits = 2
    turn.progress_callback("subagent.tool", "bash", tool_count=2, **_child(0))
    turn.progress_callback("subagent.complete", status="ok", **_child(0))
    await _drain(scheduled)
    assert len(adapter.sent) == 1 and "1/1 done" in adapter.texts[-1]


def test_overlapping_waves_use_delegation_ids_for_additive_totals():
    board = SubagentActivityBoard(None, "chat", None)
    events = [
        (0, 3, "wave-a", "a0"),
        (0, 2, "wave-b", "b0"),
        (1, 3, "wave-a", "a1"),
        (1, 2, "wave-b", "b1"),
        (2, 3, "wave-a", "a2"),
    ]
    for index, count, delegation_id, subagent_id in events:
        board.observe("subagent.start", {
            **_child(index, count), "delegation_id": delegation_id, "subagent_id": subagent_id,
        })

    assert board._render().split("\n", 1)[0] == "🔀 Subagents · 0/5 done"



@pytest.mark.asyncio
async def test_unowned_initial_delivery_is_never_retried():
    class _UnownedSend:
        def __init__(self, result):
            self.result = result
            self.attempts = 0
            self.metadata = None

        async def send(self, _chat_id, _content, *, metadata=None):
            self.attempts += 1
            self.metadata = metadata
            return self.result

    outcomes = (
        SimpleNamespace(success=False, message_id=None, retryable=False),
        SimpleNamespace(success=True, message_id=None, retryable=True),
    )
    for outcome in outcomes:
        adapter = _UnownedSend(outcome)
        metadata = {"thread_id": "77"}
        board = SubagentActivityBoard(adapter, "chat", metadata, min_edit_interval=0.0)
        metadata["thread_id"] = "changed"
        assert board.observe("subagent.start", _child(0))
        await board.run()

        assert adapter.attempts == 1
        assert adapter.metadata == {"thread_id": "77", "_interim_send": True}
        assert board._delivery_abandoned is True
        assert board.observe("subagent.complete", {**_child(0), "status": "ok"}) is False

@pytest.mark.asyncio
async def test_cancelling_an_in_flight_send_abandons_the_board():
    class _BlockingSend(_EditingAdapter):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.attempts = 0

        async def send(self, chat_id, content, *, metadata=None):
            del chat_id, content, metadata
            self.attempts += 1
            self.entered.set()
            await asyncio.Future()

    adapter = _BlockingSend()
    board = SubagentActivityBoard(adapter, "chat", None, min_edit_interval=0.0)
    assert board.observe("subagent.start", _child(0))
    publisher = asyncio.create_task(board.run())
    await adapter.entered.wait()
    publisher.cancel()
    with pytest.raises(asyncio.CancelledError):
        await publisher

    assert board._delivery_abandoned is True
    assert board._publisher_running is False
    assert board.observe("subagent.complete", {**_child(0), "status": "ok"}) is False
    assert adapter.attempts == 1


@pytest.mark.asyncio
async def test_first_send_failure_is_retried_without_leaving_a_ghost_id(scheduled):
    class _FlakySend(_EditingAdapter):
        async def send(self, chat_id, content, *, metadata=None):
            if not self.sent and not getattr(self, "_failed_once", False):
                self._failed_once = True
                return SimpleNamespace(success=False, message_id=None, retryable=True)
            return await super().send(chat_id, content, metadata=metadata)

    adapter = _FlakySend()
    turn = _turn(adapter, fake_time=_FakeTime())
    turn.progress_callback("subagent.start", **_child(0))
    await _drain(scheduled)
    turn.progress_callback("subagent.complete", status="ok", **_child(0))
    await _drain(scheduled)

    assert len(adapter.sent) == 1 and [m for m, _ in adapter.edits] == ["msg-1"]


def test_nested_grandchildren_are_ignored():
    board = SubagentActivityBoard(None, "chat", None)
    assert board.observe("subagent.start", _child(0, 2))
    # A child orchestrator relays its own children's events upward with depth >= 1.
    assert board.observe("subagent.start", {**_child(0, 3), "depth": 1}) is False
    assert board.observe("subagent.complete", {**_child(1, 3), "depth": 1, "status": "ok"}) is False
    assert "0/2 done" in board._render()
    assert board._render().count("#") == 1


def test_sequential_waves_get_fresh_ordinals_and_additive_totals():
    board = SubagentActivityBoard(None, "chat", None)
    board.observe("subagent.start", {**_child(1, 3), "subagent_id": "w1-1"})  # out of order within a wave
    board.observe("subagent.start", {**_child(0, 3), "subagent_id": "w1-0"})
    board.observe("subagent.start", {**_child(2, 3), "subagent_id": "w1-2"})
    assert "0/3 done" in board._render()
    for suffix in ("0", "1", "2"):
        board.observe("subagent.complete", {**_child(0, 3), "subagent_id": f"w1-{suffix}", "status": "ok"})
    # Second delegate_task call in the same turn: task_index restarts at 0.
    board.observe("subagent.start", {**_child(0, 1), "subagent_id": "w2-0"})
    lines = board._render().split("\n")

    assert lines[0] == "🔀 Subagents · 3/4 done"
    assert [line.split()[1] for line in lines[1:]] == ["#1", "#2", "#3", "#4"]
    assert lines[4].endswith("#4 🚀 spawned · <30s")


def test_unchanged_state_and_invisible_changes_owe_no_edit():
    fake = _FakeTime()
    board = SubagentActivityBoard(None, "chat", None, clock=fake.clock)
    assert board.observe("subagent.text", _child(0)) is False
    assert board.observe("tool.started", {}) is False
    assert board.observe("subagent.start", _child(0)) is True
    board._publisher_running = False
    assert board.observe("subagent.start", _child(0)) is False
    assert board.observe("subagent.tool", {**_child(0), "tool_count": 2}) is True
    board._publisher_running = False
    # A stale (lower) tool count from an out-of-order relay is not a change.
    assert board.observe("subagent.tool", {**_child(0), "tool_count": 1}) is False
    # A hostile tool count is clamped rather than rendered as a 300-digit number.
    board.observe("subagent.tool", {**_child(0), "tool_count": 1e300})
    assert "9999 tools" in board._render()


def test_wide_fan_out_collapses_its_tail_into_one_closing_row():
    board = SubagentActivityBoard(None, "chat", None)
    for index in range(12):
        board.observe("subagent.start", _child(index, 12))
    lines = board._render().split("\n")

    assert lines[0] == "🔀 Subagents · 0/12 done"
    assert len(lines) == 10
    assert [line[0] for line in lines[1:9]] == ["├"] * 8
    assert lines[9] == "└ …and 4 more"


@pytest.mark.asyncio
async def test_change_hidden_in_the_collapsed_tail_sends_no_edit():
    fake = _FakeTime()
    adapter = _EditingAdapter()
    board = SubagentActivityBoard(adapter, "chat", None, clock=fake.clock, sleep=fake.sleep)
    for index in range(9):
        board.observe("subagent.start", _child(index, 9))
    await board.run()
    assert board.observe("subagent.tool", {**_child(8, 9), "tool_count": 1})  # row 9 is collapsed
    await board.run()

    assert len(adapter.sent) == 1 and adapter.edits == []


@pytest.mark.parametrize(
    ("seconds", "terminal", "expected"),
    [
        (0, False, "<30s"), (29.9, False, "<30s"),
        (30, False, "<1m"), (44.9, False, "<1m"),
        (45, False, "~1m"), (89.9, False, "~1m"), (90, False, "~2m"),
        (449, False, "~7m"), (450, False, "~8m"),
        (3569, False, "~59m"), (3570, False, "~1h00m"),
        (4049, False, "~1h07m"), (4050, False, "~1h08m"),
        (93599, False, "~1d02h"), (93600, False, "~1d02h"),
        (0, True, "0s"), (59.9, True, "59s"), (65, True, "1m05s"),
        (457, True, "7m37s"), (4020, True, "1h07m"), (93600, True, "1d02h"),
    ],
)
def test_elapsed_precision_matches_activity_state(seconds, terminal, expected):
    assert _format_elapsed(seconds, terminal=terminal) == expected


def test_heartbeat_advances_running_elapsed_and_completion_freezes_it():
    fake = _FakeTime()
    board = SubagentActivityBoard(None, "chat", None, clock=fake.clock)
    board.observe("subagent.start", _child(0))
    board.publisher_not_started()
    fake.now += 449
    assert board.observe("subagent.heartbeat", _child(0))
    assert "🚀 spawned · ~7m" in board._render()

    board.observe("subagent.complete", {**_child(0), "status": "ok", "duration_seconds": 125})
    fake.now += 3600
    assert board.observe("subagent.heartbeat", _child(0)) is False
    assert "✅ done · 2m05s" in board._render()


def test_display_setting_defaults_on_and_can_be_switched_off_per_platform():
    assert resolve_display_setting({}, "telegram", "subagent_activity_board") is True
    assert resolve_display_setting({}, "discord", "subagent_activity_board") is False
    config = {"display": {"platforms": {"telegram": {"subagent_activity_board": "off"}}}}
    assert resolve_display_setting(config, "telegram", "subagent_activity_board") is False
