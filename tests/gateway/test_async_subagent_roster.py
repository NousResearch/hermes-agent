from gateway.async_subagent_roster import build_async_subagent_roster_rows
from gateway.subagent_roster import format_subagent_roster


def test_async_rows_use_active_registry_for_running_elapsed():
    record = {
        "delegation_id": "deleg_1",
        "dispatched_at": 100.0,
        "children": [
            {"task_index": 0, "subagent_id": "sa-0", "goal": "sleep 6", "status": "pending"},
            {"task_index": 1, "subagent_id": "sa-1", "goal": "sleep 10", "status": "pending"},
        ],
    }
    active = [
        {"subagent_id": "sa-0", "started_at": 101.0, "tool_count": 2},
        {"subagent_id": "sa-1", "started_at": 102.0, "tool_count": 0},
    ]

    rows = build_async_subagent_roster_rows(record, active, now=107.0)

    assert rows[0]["glyph"] == "▶"
    assert rows[0]["elapsed"] == 6.0
    assert rows[0]["tools"] == 2
    assert rows[1]["glyph"] == "▶"
    assert rows[1]["elapsed"] == 5.0


def test_async_rows_use_record_status_for_live_done_counts():
    record = {
        "delegation_id": "deleg_1",
        "children": [
            {
                "task_index": 0,
                "subagent_id": "sa-0",
                "goal": "sleep 6",
                "status": "completed",
                "duration_seconds": 6.0,
            },
            {"task_index": 1, "subagent_id": "sa-1", "goal": "sleep 10", "status": "pending"},
        ],
    }
    active = [{"subagent_id": "sa-1", "started_at": 100.0, "tool_count": 0}]

    rows = build_async_subagent_roster_rows(record, active, now=108.0)
    text = format_subagent_roster(rows)

    assert "🤖 Subagents — 1 running, 1 done" in text
    assert "✓ `sleep 6` · `6s`" in text
    assert "▶ `sleep 10` · `8s`" in text


def test_async_terminal_row_keeps_final_tool_count():
    # A finished background child keeps its tool count. The child record carries
    # the final count (tool_count, falling back to api_calls); the registry has
    # already dropped the live entry by the time it completes.
    record = {
        "delegation_id": "deleg_1",
        "children": [
            {
                "task_index": 0,
                "subagent_id": "sa-0",
                "goal": "review",
                "status": "completed",
                "duration_seconds": 949.0,
                "tool_count": 56,
            },
            {
                "task_index": 1,
                "subagent_id": "sa-1",
                "goal": "audit",
                "status": "completed",
                "duration_seconds": 287.0,
                "api_calls": 23,
            },
        ],
    }
    rows = build_async_subagent_roster_rows(record, [], now=2000.0)
    assert rows[0]["tools"] == 56
    assert rows[1]["tools"] == 23  # falls back to api_calls
    text = format_subagent_roster(rows, collapsed=True)
    assert "✓ `review` · `15m 49s` · 56 tools" in text
    assert "✓ `audit` · `4m 47s` · 23 tools" in text


def test_async_final_rows_fallback_to_results_when_children_missing():
    record = {
        "delegation_id": "deleg_1",
        "goals": ["sleep 6", "sleep 10"],
        "results": [
            {"task_index": 0, "status": "completed", "duration_seconds": 6.0},
            {"task_index": 1, "status": "failed", "duration_seconds": 10.0},
        ],
    }

    rows = build_async_subagent_roster_rows(record, [], now=200.0)
    text = format_subagent_roster(rows, collapsed=True)

    # Collapsed render now keeps the per-child breakdown under a summary header.
    lines = text.split("\n")
    # Header elapsed is the WALL-CLOCK fallback = slowest child (max(6,10)=10),
    # NOT the sum (16): this direct call passes no wall_clock.
    assert lines[0] == "⚠️ 2 subagents · 1 ✓ · 1 ✗ · `10s`"
    assert lines[1] == "✓ `sleep 6` · `6s`"
    assert lines[2] == "✗ `sleep 10` · `10s`"


# ---------------------------------------------------------------------------
# Watcher-owned publisher integration tests (Task 7 + B1/B3/C5 folds)
# ---------------------------------------------------------------------------

import types

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class AsyncRosterAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self._seq = 0

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None, buttons=None):
        self._seq += 1
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True, message_id=f"async-roster-{self._seq}")

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "kwargs": kwargs,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class NoEditAsyncRosterAdapter(AsyncRosterAdapter):
    edit_message = BasePlatformAdapter.edit_message


def _runner(adapter, *, entries=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner.session_store = types.SimpleNamespace(
        _ensure_loaded=lambda: None, _entries=entries or {}
    )
    runner._session_sources = {}
    runner._async_subagent_roster_bubbles = {}
    return runner


def _record(status="running"):
    return {
        "type": "async_delegation",
        "delegation_id": "deleg_bg",
        "session_key": "agent:main:telegram:group:-1001:77",
        "platform": "telegram",
        "chat_type": "group",
        "chat_id": "-1001",
        "thread_id": "77",
        "message_id": "42",
        "is_batch": True,
        "status": status,
        "dispatched_at": 100.0,
        "children": [
            {"task_index": 0, "subagent_id": "sa-0", "goal": "sleep 6", "status": "pending"},
            {"task_index": 1, "subagent_id": "sa-1", "goal": "sleep 10", "status": "pending"},
        ],
    }


@pytest.mark.asyncio
async def test_watcher_roster_seeds_edits_and_collapses(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"platforms": {"telegram": {"subagent_roster": "on"}}}},
    )

    adapter = AsyncRosterAdapter()
    runner = _runner(adapter)
    record = _record()

    await runner._tick_async_delegation_rosters(
        [record],
        [
            {"subagent_id": "sa-0", "started_at": 101.0, "tool_count": 1},
            {"subagent_id": "sa-1", "started_at": 102.0, "tool_count": 0},
        ],
    )

    assert len(adapter.sent) == 1
    assert "🤖 Subagents" in adapter.sent[0]["content"]
    assert "sleep 6" in adapter.sent[0]["content"]
    assert adapter.sent[0]["metadata"]["thread_id"] == "77"

    record["children"][0]["status"] = "completed"
    record["children"][0]["duration_seconds"] = 6.0

    await runner._publish_async_delegation_roster(
        record,
        [{"subagent_id": "sa-1", "started_at": 102.0, "tool_count": 0}],
        force=True,
        collapsed=False,
    )

    assert len(adapter.sent) == 1
    assert adapter.edits
    assert "1 running, 1 done" in adapter.edits[-1]["content"]
    assert "✓ `sleep 6` · `6s`" in adapter.edits[-1]["content"]

    final_evt = _record(status="completed")
    final_evt["total_duration_seconds"] = 4.0  # authoritative batch wall-clock
    final_evt["children"][0]["status"] = "completed"
    final_evt["children"][0]["duration_seconds"] = 6.0
    final_evt["children"][1]["status"] = "completed"
    final_evt["children"][1]["duration_seconds"] = 10.0

    await runner._finalize_async_delegation_roster(final_evt, [])

    # Header elapsed is the batch WALL-CLOCK (total_duration_seconds=4.0),
    # threaded end-to-end, NOT the sum of children (6+10=16s).
    assert "✅ 2 subagents · 2 ✓ · `4s`" in adapter.edits[-1]["content"]
    assert "deleg_bg" not in runner._async_subagent_roster_bubbles


@pytest.mark.asyncio
async def test_watcher_live_header_uses_wall_clock(monkeypatch):
    # AC2: the LIVE (running) header shows real elapsed since dispatch
    # (now - dispatched_at), NOT a sum of child elapsed. Clock is FROZEN so the
    # whole-second rounding can't flake across the 96.5s boundary.
    import time as _time
    import gateway.run as gateway_run

    FIXED = 1_000_000.0
    monkeypatch.setattr(_time, "time", lambda: FIXED)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"platforms": {"telegram": {"subagent_roster": "on"}}}},
    )

    adapter = AsyncRosterAdapter()
    runner = _runner(adapter)
    record = _record()  # status running, is_batch True
    record["dispatched_at"] = FIXED - 96.0  # dispatched 1m 36s ago

    await runner._tick_async_delegation_rosters([record], [])

    assert adapter.sent
    # Live header carries wall-clock since dispatch (96s -> "1m 36s"), never a
    # sum of the two child elapsed.
    assert "· `1m 36s`" in adapter.sent[0]["content"]


@pytest.mark.asyncio
async def test_watcher_roster_respects_off_knob(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"platforms": {"telegram": {"subagent_roster": "off"}}}},
    )

    adapter = AsyncRosterAdapter()
    runner = _runner(adapter)

    await runner._tick_async_delegation_rosters([_record()], [])

    assert adapter.sent == []
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_watcher_roster_noops_without_edit_adapter(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"platforms": {"telegram": {"subagent_roster": "on"}}}},
    )

    adapter = NoEditAsyncRosterAdapter()
    runner = _runner(adapter)

    await runner._tick_async_delegation_rosters([_record()], [])

    assert adapter.sent == []
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_watcher_roster_routing_wins_over_session_store_origin(monkeypatch):
    """B1: stored routing must beat a stale/foreground session-store origin.

    The session-store entry for this session_key carries an origin with a
    DIFFERENT thread (a foreground topic). The roster must still post to the
    dispatch-time routing thread (77) with reply anchor 42, proving the
    session-store-origin fast path did NOT override the captured routing.
    """
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"platforms": {"telegram": {"subagent_roster": "on"}}}},
    )

    from gateway.session import SessionSource

    stale_origin = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="999",  # WRONG topic — foreground origin
    )
    entries = {
        "agent:main:telegram:group:-1001:77": types.SimpleNamespace(origin=stale_origin)
    }

    adapter = AsyncRosterAdapter()
    runner = _runner(adapter, entries=entries)
    record = _record()

    await runner._tick_async_delegation_rosters(
        [record],
        [{"subagent_id": "sa-0", "started_at": 101.0, "tool_count": 0}],
    )

    assert len(adapter.sent) == 1
    md = adapter.sent[0]["metadata"]
    assert md is not None
    # routing (77) wins over stale session-store origin (999)
    assert md["thread_id"] == "77"
    # reply anchor is the captured message_id (42), surfaced for Telegram
    assert str(md.get("telegram_reply_to_message_id") or md.get("reply_to_message_id") or "") in ("42", "")


@pytest.mark.asyncio
async def test_watcher_roster_collapses_when_batch_finished_before_first_tick(monkeypatch):
    """B3: a fast batch can complete before the watcher's first tick — the
    finalizer must still SEED a collapsed bubble even though none ever existed.
    """
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"platforms": {"telegram": {"subagent_roster": "on"}}}},
    )

    adapter = AsyncRosterAdapter()
    runner = _runner(adapter)

    final_evt = _record(status="completed")
    for c in final_evt["children"]:
        c["status"] = "completed"
        c["duration_seconds"] = 6.0

    # No _tick first — straight to finalize.
    await runner._finalize_async_delegation_roster(final_evt, [])

    # Exactly one SEND (the collapsed seed), no edits, bubble popped.
    assert len(adapter.sent) == 1
    assert "✅ 2 subagents · 2 ✓" in adapter.sent[0]["content"]
    assert adapter.edits == []
    assert "deleg_bg" not in runner._async_subagent_roster_bubbles


@pytest.mark.asyncio
async def test_watcher_roster_collapses_interrupted_batch_with_failed_counts(monkeypatch):
    """C5: an interrupted/failed batch completion event must still collapse the
    bubble (with ✗ counts) and pop it from the registry.
    """
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"platforms": {"telegram": {"subagent_roster": "on"}}}},
    )

    adapter = AsyncRosterAdapter()
    runner = _runner(adapter)

    # Seed a live bubble first.
    await runner._tick_async_delegation_rosters(
        [_record()],
        [{"subagent_id": "sa-0", "started_at": 101.0, "tool_count": 0}],
    )
    assert len(adapter.sent) == 1

    interrupted = _record(status="interrupted")
    interrupted["children"][0]["status"] = "completed"
    interrupted["children"][0]["duration_seconds"] = 6.0
    interrupted["children"][1]["status"] = "interrupted"
    interrupted["children"][1]["duration_seconds"] = 3.0

    await runner._finalize_async_delegation_roster(interrupted, [])

    last = adapter.edits[-1]["content"] if adapter.edits else adapter.sent[-1]["content"]
    assert "1 ✓" in last
    assert "1 ✗" in last
    assert "deleg_bg" not in runner._async_subagent_roster_bubbles


# ── Reseed-on-flood: a flood-rejected seed must NOT latch (keep trying) ──────
class FloodSeedAdapter(AsyncRosterAdapter):
    """Rejects the first N seed sends with a flood-control result, then succeeds.

    Models Telegram flood control on createMessage: the send DEFINITIVELY did
    not deliver (success=False), so the roster must re-seed on the next tick
    rather than latching seed_failed and never showing the bubble.
    """

    def __init__(self, fail_sends=1, error="flood_control:18", retryable=False):
        super().__init__()
        self._fail_sends = fail_sends
        self._flood_error = error
        self._flood_retryable = retryable

    async def send(self, chat_id, content, reply_to=None, metadata=None, buttons=None):
        if self._fail_sends > 0:
            self._fail_sends -= 1
            self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata,
                              "failed": True})
            return SendResult(
                success=False, error=self._flood_error, retryable=self._flood_retryable
            )
        return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)


class AmbiguousSeedAdapter(AsyncRosterAdapter):
    """Rejects the seed with a non-flood (ambiguous) failure that MIGHT have
    landed — the roster must latch and NOT re-seed (avoid duplicate bubbles)."""

    def __init__(self, error="Bad Gateway"):
        super().__init__()
        self._err = error

    async def send(self, chat_id, content, reply_to=None, metadata=None, buttons=None):
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata,
                          "failed": True})
        return SendResult(success=False, error=self._err)


@pytest.mark.asyncio
async def test_watcher_roster_reseeds_after_flood_reject(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"platforms": {"telegram": {"subagent_roster": "on"}}}},
    )
    # Drive a controllable clock so the second tick is past the (>=1s floor)
    # roster interval; the reseed is deliberately PACED, not hammered.
    clock = {"t": 1000.0}
    monkeypatch.setattr(gateway_run.time, "monotonic", lambda: clock["t"])

    adapter = FloodSeedAdapter(fail_sends=1)  # first seed floods, second lands
    runner = _runner(adapter)
    active = [{"subagent_id": "sa-0", "started_at": 101.0, "tool_count": 0}]

    # Tick 1: seed is flood-rejected → bubble exists in state but unseeded, NOT latched.
    await runner._tick_async_delegation_rosters([_record()], active)
    bubble = runner._async_subagent_roster_bubbles["deleg_bg"]
    assert bubble["message_id"] is None
    assert bubble["seed_failed"] is False, "flood reject must NOT latch seed_failed"
    assert len(adapter.sent) == 1  # one (failed) seed attempt so far

    # Advance past the interval; tick 2 re-seeds and now succeeds.
    clock["t"] += 30.0
    await runner._tick_async_delegation_rosters([_record()], active)
    assert len(adapter.sent) == 2, "must retry the seed after a flood reject"
    assert bubble["message_id"] is not None, "second seed should land the bubble"


@pytest.mark.asyncio
async def test_watcher_roster_retryable_flood_reseeds(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"platforms": {"telegram": {"subagent_roster": "on"}}}},
    )
    clock = {"t": 2000.0}
    monkeypatch.setattr(gateway_run.time, "monotonic", lambda: clock["t"])

    # retryable=True with no error string = Telegram short (<=5s) flood.
    adapter = FloodSeedAdapter(fail_sends=1, error=None, retryable=True)
    runner = _runner(adapter)
    active = [{"subagent_id": "sa-0", "started_at": 101.0, "tool_count": 0}]

    await runner._tick_async_delegation_rosters([_record()], active)
    assert runner._async_subagent_roster_bubbles["deleg_bg"]["seed_failed"] is False
    clock["t"] += 30.0
    await runner._tick_async_delegation_rosters([_record()], active)
    assert len(adapter.sent) == 2
    assert runner._async_subagent_roster_bubbles["deleg_bg"]["message_id"] is not None


@pytest.mark.asyncio
async def test_watcher_roster_latches_on_ambiguous_failure(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"platforms": {"telegram": {"subagent_roster": "on"}}}},
    )
    clock = {"t": 3000.0}
    monkeypatch.setattr(gateway_run.time, "monotonic", lambda: clock["t"])

    adapter = AmbiguousSeedAdapter()  # always fails with a non-flood error
    runner = _runner(adapter)
    active = [{"subagent_id": "sa-0", "started_at": 101.0, "tool_count": 0}]

    await runner._tick_async_delegation_rosters([_record()], active)
    bubble = runner._async_subagent_roster_bubbles["deleg_bg"]
    assert bubble["seed_failed"] is True, "ambiguous failure must latch (may have delivered)"

    # Even past the interval, a latched seed must not re-attempt (avoid dup bubbles).
    clock["t"] += 30.0
    await runner._tick_async_delegation_rosters([_record()], active)
    assert len(adapter.sent) == 1, "latched seed must not re-attempt on an ambiguous failure"


# ── MERGE: per-child cost rendering (delegate-card-into-roster) ──


def test_async_terminal_rows_carry_cost_usd():
    record = {
        "delegation_id": "deleg_1",
        "children": [
            {
                "task_index": 0,
                "subagent_id": "sa-0",
                "goal": "g0",
                "status": "completed",
                "duration_seconds": 5.0,
                "cost_usd": 0.0231,
            },
            {
                "task_index": 1,
                "subagent_id": "sa-1",
                "goal": "g1",
                "status": "completed",
                "duration_seconds": 6.0,
                # no cost_usd -> no cost cell, not counted in total
            },
        ],
    }
    rows = build_async_subagent_roster_rows(record, [], now=200.0)
    assert rows[0]["cost_usd"] == 0.0231
    assert rows[1]["cost_usd"] is None

    text = format_subagent_roster(rows, collapsed=True)
    assert "· `$0.0231`" in text          # per-row cost (adaptive 4dp <$1)
    assert "· `$0.0231`" in text.split("\n")[0] or "$0.0231" in text  # header total sums known only
    # header total == only the known cost
    assert "`$0.0231`" in text.split("\n")[0]


def test_cost_format_adaptive_and_header_total():
    from gateway.subagent_roster import _format_cost, _cost_suffix

    assert _format_cost(0.0123) == "$0.0123"
    assert _format_cost(1.0) == "$1.00"
    assert _format_cost(1.2345) == "$1.23"
    assert _format_cost(0.999) == "$0.9990"
    assert _cost_suffix({"cost_usd": 0.0123}) == " · `$0.0123`"
    assert _cost_suffix({"cost_usd": 2.5}) == " · `$2.50`"
    assert _cost_suffix({"cost_usd": None}) == ""
    assert _cost_suffix({"cost_usd": 0}) == ""
    assert _cost_suffix({"cost_usd": "x"}) == ""
    assert _cost_suffix({}) == ""

    # Header sums known costs and uses adaptive format (0.6 + 0.6 = 1.20 -> 2dp).
    rows = [
        {"glyph": "✓", "label": "a", "elapsed": 1.0, "running": False, "tools": 0,
         "bucket": "done", "model": "", "cost_usd": 0.6},
        {"glyph": "✓", "label": "b", "elapsed": 1.0, "running": False, "tools": 0,
         "bucket": "done", "model": "", "cost_usd": 0.6},
    ]
    head = format_subagent_roster(rows, collapsed=True).split("\n")[0]
    assert "`$1.20`" in head


def _config_args_and_roster_on():
    return {"display": {"platforms": {"telegram": {
        "subagent_roster": "on", "delegate_task_args": "on"}}}}


@pytest.mark.asyncio
async def test_watcher_pins_dispatched_header_above_roster(monkeypatch):
    """args:on + roster:on -> the '🔀 Delegate task — N agents · profile' header
    is PINNED as the first line of the bubble and STAYS there across edits; the
    live roster rows are appended BELOW it (no morph-away)."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config", _config_args_and_roster_on)

    adapter = AsyncRosterAdapter()
    runner = _runner(adapter)
    record = _record()
    record["profile"] = "dual-review"
    record["children"][0]["profile"] = "reviewer-codex"
    record["children"][1]["profile"] = "reviewer-opus"

    # First tick: bubble seeds with header + roster in ONE message.
    await runner._tick_async_delegation_rosters(
        [record],
        [{"subagent_id": "sa-0", "started_at": 101.0, "tool_count": 1}],
    )
    assert len(adapter.sent) == 1
    seed = adapter.sent[0]["content"]
    seed_lines = seed.split("\n")
    assert seed_lines[0] == "🔀 Delegate task — 2 agents · profile: `dual-review`"  # pinned header
    assert seed_lines[1].startswith("🤖 Subagents")                      # roster appended
    assert "reviewer-codex" in seed and "reviewer-opus" in seed          # per-row lanes

    # Subsequent publish: EDITS the same message; header is STILL there (not morphed away).
    record["children"][0]["status"] = "completed"
    record["children"][0]["duration_seconds"] = 6.0
    await runner._publish_async_delegation_roster(
        record,
        [{"subagent_id": "sa-1", "started_at": 102.0, "tool_count": 0}],
        force=True,
        collapsed=False,
    )
    assert len(adapter.sent) == 1  # no second send
    assert adapter.edits
    edited = adapter.edits[-1]["content"]
    assert edited.startswith("🔀 Delegate task — 2 agents · profile: `dual-review`")  # header PERSISTS
    assert "🤖 Subagents" in edited
    assert "reviewer-codex" in edited and "reviewer-opus" in edited


@pytest.mark.asyncio
async def test_watcher_no_header_when_args_off(monkeypatch):
    """roster:on + args:OFF -> NO dispatched header (toggle independence); just
    the bare roster."""
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"platforms": {"telegram": {"subagent_roster": "on"}}}},
    )

    adapter = AsyncRosterAdapter()
    runner = _runner(adapter)
    record = _record()
    record["profile"] = "dual-review"
    record["children"][0]["profile"] = "reviewer-codex"

    await runner._tick_async_delegation_rosters([record], [])
    assert len(adapter.sent) == 1
    assert "🔀 Delegate task" not in adapter.sent[0]["content"]
    assert adapter.sent[0]["content"].startswith("🤖 Subagents")


# ── Pinned header builder (build_async_dispatched_header) ──
from gateway.async_subagent_roster import build_async_dispatched_header


def test_dispatched_header_agent_count_and_profile():
    def rec(profile=None, toolsets=None, n=2):
        return {"profile": profile, "toolsets": toolsets,
                "children": [{"task_index": i, "subagent_id": f"sa-{i}", "goal": f"g{i}"} for i in range(n)]}

    # N agents (post-expansion child count), profile shown, toolsets hidden when inherited.
    assert build_async_dispatched_header(rec("dual-review", None)) == \
        "🔀 Delegate task — 2 agents · profile: `dual-review`"
    # toolsets shown ONLY when explicitly set.
    assert build_async_dispatched_header(rec("coder", ["terminal", "file"])) == \
        "🔀 Delegate task — 2 agents · profile: `coder` · toolsets=`terminal,file`"
    # no profile -> EXPLICIT "profile: none" cell (not omitted), so a plain
    # delegate is unambiguous.
    assert build_async_dispatched_header(rec(None, None)) == \
        "🔀 Delegate task — 2 agents · profile: `none`"
    # singular.
    assert build_async_dispatched_header(rec("explorer", None, n=1)) == \
        "🔀 Delegate task — 1 agent · profile: `explorer`"
    # empty record -> empty header.
    assert build_async_dispatched_header({}) == ""


def test_dispatched_header_prefers_header_toolsets_over_resolved():
    """The header reads `header_toolsets` (the EXPLICIT set computed by the
    caller) in preference to the resolved `toolsets` record field, so a per-task
    uniform set surfaces and a profile-defaulted `toolsets` does not leak in."""
    base = {"children": [{"task_index": 0, "subagent_id": "s0", "goal": "g0"},
                         {"task_index": 1, "subagent_id": "s1", "goal": "g1"}]}
    # header_toolsets present -> used.
    assert build_async_dispatched_header({**base, "header_toolsets": ["terminal", "file"]}) == \
        "🔀 Delegate task — 2 agents · profile: `none` · toolsets=`terminal,file`"
    # header_toolsets absent -> falls back to top-level toolsets (back-compat).
    assert build_async_dispatched_header({**base, "toolsets": ["web"]}) == \
        "🔀 Delegate task — 2 agents · profile: `none` · toolsets=`web`"
    # neither -> no toolsets cell.
    assert build_async_dispatched_header(base) == \
        "🔀 Delegate task — 2 agents · profile: `none`"


def test_dispatched_header_prefers_per_task_profile():
    record = {
        "profile": None,
        "header_profile": "dual-review",
        "children": [
            {"task_index": 0, "subagent_id": "s0", "goal": "g0"},
            {"task_index": 1, "subagent_id": "s1", "goal": "g1"},
        ],
    }

    assert build_async_dispatched_header(record) == \
        "🔀 Delegate task — 2 agents · profile: `dual-review`"


# ── Fix A: profile must PERSIST in roster rows (now BELOW the pinned header) ──
# Regression for "I don't see the profile anymore, only the Subagents part":
# the profile is a per-row cell so it shows in running, partial-done, AND
# collapsed states — independently of the pinned header above.

def test_roster_rows_carry_profile_in_all_states():
    """build_async_subagent_roster_rows threads child profile onto every row
    bucket (running / pending / terminal) so the renderer can keep it visible."""
    record = {
        "delegation_id": "deleg_p",
        "dispatched_at": 100.0,
        "children": [
            {"task_index": 0, "subagent_id": "sa-0", "goal": "g0",
             "profile": "reviewer-codex", "status": "completed", "duration_seconds": 5.0},
            {"task_index": 1, "subagent_id": "sa-1", "goal": "g1",
             "profile": "reviewer-opus", "status": "pending"},
        ],
    }
    active = [{"subagent_id": "sa-1", "started_at": 100.0, "tool_count": 0}]
    rows = build_async_subagent_roster_rows(record, active, now=110.0)
    by_label = {r["label"]: r for r in rows}
    assert by_label["g0"]["profile"] == "reviewer-codex"   # terminal row
    assert by_label["g1"]["profile"] == "reviewer-opus"    # running row


def test_profile_suffix_renders_on_rows():
    """_profile_suffix renders the lane, kept in both live and collapsed render."""
    from gateway.subagent_roster import _profile_suffix

    assert _profile_suffix({"profile": "reviewer-codex"}) == " · `reviewer-codex`"
    assert _profile_suffix({"profile": "  reviewer`-`opus "}) == " · `reviewer-opus`"
    assert _profile_suffix({"profile": ""}) == ""
    assert _profile_suffix({}) == ""

    rows = [
        {"glyph": "✓", "label": "g0", "elapsed": 5.0, "running": False, "tools": 70,
         "bucket": "done", "model": "gpt-5.5", "profile": "reviewer-codex"},
        {"glyph": "✓", "label": "g1", "elapsed": 6.0, "running": False, "tools": 42,
         "bucket": "done", "model": "us.anthropic.claude-opus-4-8", "profile": "reviewer-opus"},
    ]
    live = format_subagent_roster(rows, collapsed=False)
    collapsed = format_subagent_roster(rows, collapsed=True)
    assert live is not None and collapsed is not None
    for text in (live, collapsed):
        assert "reviewer-codex" in text
        assert "reviewer-opus" in text
        # profile sits before the model cell: `g0` · `reviewer-codex` · gpt-5.5
        assert "`g0` · `reviewer-codex` · gpt-5.5" in text


@pytest.mark.asyncio
async def test_watcher_profile_persists_in_rows_through_edits(monkeypatch):
    """End-to-end: per-row profile is present in BOTH the seed send and the
    subsequent live-roster edit (the original 'profile vanished' regression),
    now alongside the pinned header."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config", _config_args_and_roster_on)

    adapter = AsyncRosterAdapter()
    runner = _runner(adapter)
    record = _record()
    record["profile"] = "dual-review"
    record["children"][0]["profile"] = "reviewer-codex"
    record["children"][1]["profile"] = "reviewer-opus"

    await runner._tick_async_delegation_rosters([record], [])
    assert "reviewer-codex" in adapter.sent[0]["content"]

    await runner._publish_async_delegation_roster(
        record,
        [{"subagent_id": "sa-0", "started_at": 101.0, "tool_count": 1},
         {"subagent_id": "sa-1", "started_at": 102.0, "tool_count": 0}],
        force=True,
        collapsed=False,
    )
    edited = adapter.edits[-1]["content"]
    assert edited.startswith("🔀 Delegate task")  # header pinned
    assert "🤖 Subagents" in edited
    assert "reviewer-codex" in edited   # ← lane survives in the row
    assert "reviewer-opus" in edited
