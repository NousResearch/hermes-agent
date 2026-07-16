"""Tests for the Slack-native task card stream (PR #59010).

Two test surfaces, matching the review asks:

1. **Callback hygiene (AST invariant)** — ``agent.reasoning_callback`` must
   be assigned on EVERY turn in gateway/run.py's per-message callback block,
   including ``None`` when native cards are inactive. Cached agents are
   reused across turns and ``_init_cached_agent_for_turn()`` does not clear
   callbacks, so a conditional-only assignment would leave the previous
   turn's closure active after the feature is toggled off — reasoning
   deltas would append to a stopped, stale task stream. Pinned the same way
   as test_10710_auto_reset_evicts_cached_agent.py (load-bearing AST pin).

2. **Fake-client stream lifecycle** — SlackTaskStream driven end-to-end
   against a fake Slack client: enablement/fallback (open failure → cards
   disabled, later events no-op), ordered completion (started → finished
   renders in_progress → complete with one card per task id), proactive +
   reactive rollover with in-progress replay, and subagent cards.
"""
from __future__ import annotations

import ast
import asyncio
import inspect

from gateway.slack_task_stream import SlackTaskStream


# ---------------------------------------------------------------------------
# 1. Callback hygiene — reasoning_callback assigned every turn
# ---------------------------------------------------------------------------


def test_reasoning_callback_assigned_unconditionally_every_turn():
    """The per-message callback block must assign ``agent.reasoning_callback``
    as a plain (non-branching) statement — a conditional *expression* with an
    explicit ``else None`` is the accepted shape, an ``if:`` *statement*
    guarding the assignment is not. This is what guarantees a cached agent
    reused on a turn with cards disabled has the stale closure cleared."""
    from gateway import run as gateway_run

    tree = ast.parse(inspect.getsource(gateway_run))
    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute)
            and t.attr == "reasoning_callback"
            and isinstance(t.value, ast.Name)
            and t.value.id == "agent"
            for t in node.targets
        )
    ]
    assert assigns, "agent.reasoning_callback assignment not found in gateway/run.py"

    # Every assignment must carry a None-producing branch: either a
    # conditional expression (IfExp) whose orelse is None, or a direct
    # None assignment. A bare closure assignment (no None branch anywhere)
    # means some path can leave a stale closure on the cached agent.
    def _has_none_branch(node: ast.Assign) -> bool:
        v = node.value
        if isinstance(v, ast.Constant) and v.value is None:
            return True
        if isinstance(v, ast.IfExp):
            return isinstance(v.orelse, ast.Constant) and v.orelse.value is None
        return False

    assert any(_has_none_branch(a) for a in assigns), (
        "agent.reasoning_callback must be assigned with an explicit None "
        "branch (e.g. `x if cards_active else None`) so cached agents are "
        "cleared on turns where native cards are inactive"
    )

    # And it must NOT live under an `if` statement that would skip the
    # assignment entirely on the disabled path (the original bug shape:
    # `if _slack_native_cards ...: agent.reasoning_callback = ...`).
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Assign)
                    and any(
                        isinstance(t, ast.Attribute)
                        and t.attr == "reasoning_callback"
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "agent"
                        for t in sub.targets
                    )
                    and not node.orelse
                    # Only flag the feature-gate shape; unrelated defensive
                    # `if agent is not None:` wrappers are fine because they
                    # gate on agent existence, not feature toggles.
                    and any(
                        isinstance(n, ast.Name)
                        and ("slack" in n.id.lower() or "card" in n.id.lower())
                        for n in ast.walk(node.test)
                    )
                ):
                    raise AssertionError(
                        "agent.reasoning_callback is assigned inside a "
                        "feature-gated `if` with no else branch — cached "
                        "agents keep the stale closure when the gate is off "
                        f"(line {node.lineno})"
                    )


def test_cached_agent_toggle_clears_stale_closure():
    """Behavioral twin of the AST pin: simulate two turns on one cached
    agent — cards ON then cards OFF — using the same assignment expression
    shape gateway/run.py uses. After the OFF turn the stale closure must be
    gone, so a reasoning delta fired mid-turn reaches nothing."""
    from types import SimpleNamespace

    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent.verbose_logging = False

    captured: list[str] = []

    def _slack_reasoning_event(text: str) -> None:
        captured.append(text)

    # Turn 1: native cards active.
    _slack_native_cards, _slack_task_stream = True, object()
    agent.reasoning_callback = (
        _slack_reasoning_event
        if (_slack_native_cards and _slack_task_stream is not None)
        else None
    )
    agent._fire_reasoning_delta("turn one thinking")
    assert captured == ["turn one thinking"]

    # Turn 2 on the SAME cached agent: feature toggled off.
    _slack_native_cards, _slack_task_stream = False, None
    agent.reasoning_callback = (
        _slack_reasoning_event
        if (_slack_native_cards and _slack_task_stream is not None)
        else None
    )
    agent._fire_reasoning_delta("turn two thinking")
    assert captured == ["turn one thinking"], (
        "stale closure fired on the cards-off turn — cached agent kept the "
        "previous turn's reasoning_callback"
    )


# ---------------------------------------------------------------------------
# 2. Fake-client stream lifecycle
# ---------------------------------------------------------------------------


class FakeSlackClient:
    """Minimal async Slack Web API double for the stream endpoints.

    Records every call; failure modes are scripted per-test:
      * ``fail_start`` — chat.startStream raises (enablement/fallback)
      * ``fail_appends_with`` — appendStream raises this error string once,
        then succeeds (reactive rollover)
    """

    def __init__(self, fail_start: bool = False):
        self.fail_start = fail_start
        self.fail_appends_with: str | None = None
        self.calls: list[tuple[str, dict]] = []
        self._ts_counter = 0

    async def chat_startStream(self, **kwargs):
        self.calls.append(("startStream", kwargs))
        if self.fail_start:
            raise RuntimeError("scripted startStream failure")
        self._ts_counter += 1
        return {"ts": f"171000000{self._ts_counter}.000"}

    async def chat_appendStream(self, **kwargs):
        self.calls.append(("appendStream", kwargs))
        if self.fail_appends_with:
            err, self.fail_appends_with = self.fail_appends_with, None
            raise RuntimeError(err)
        return {"ok": True}

    async def chat_stopStream(self, **kwargs):
        self.calls.append(("stopStream", kwargs))
        return {"ok": True}

    # -- inspection helpers -------------------------------------------------
    def appended_chunks(self) -> list[dict]:
        return [
            chunk
            for name, kw in self.calls
            if name == "appendStream"
            for chunk in kw.get("chunks", [])
        ]

    def stream_opens(self) -> int:
        return sum(1 for name, _ in self.calls if name == "startStream")


def _make_stream(client: FakeSlackClient, **kw) -> SlackTaskStream:
    return SlackTaskStream(
        client=client,
        channel="C123",
        thread_ts="1710000000.100",
        recipient_team_id="T1",
        recipient_user_id="U1",
        **kw,
    )


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class TestEnablementFallback:
    def test_start_failure_disables_cards_and_later_events_noop(self):
        client = FakeSlackClient(fail_start=True)
        stream = _make_stream(client)

        async def scenario():
            await stream.task_started(1, "terminal", preview="ls")
            assert stream.disabled is True
            # Subsequent events must be no-ops (no retry storm on a broken stream).
            await stream.task_finished(1, "terminal", duration=0.5)
            await stream.reasoning_update("thinking that goes nowhere")
            await stream.subagent_event("subagent.start", "k1", goal="g")
            await stream.stop()

        _run(scenario())
        assert client.stream_opens() == 1  # exactly one attempt, no retries
        assert client.appended_chunks() == []
        # stop() on a disabled stream must not call stopStream.
        assert all(name != "stopStream" for name, _ in client.calls)

    def test_append_hard_failure_disables_for_rest_of_turn(self):
        client = FakeSlackClient()
        stream = _make_stream(client)
        client_calls_before = 0

        async def scenario():
            nonlocal client_calls_before
            await stream.task_started(1, "terminal", preview="ls")
            client_calls_before = len(client.calls)
            # Non-recoverable append error (not a rollover-able one).
            client.fail_appends_with = "some_fatal_slack_error"
            await stream.task_started(2, "web_search", preview="q")
            assert stream.disabled is True
            await stream.task_finished(2, "web_search")  # must no-op

        _run(scenario())
        assert len(client.calls) == client_calls_before + 1  # the failed append only


class TestOrderedCompletion:
    def test_started_then_finished_same_card_in_order(self):
        client = FakeSlackClient()
        stream = _make_stream(client)

        async def scenario():
            await stream.task_started(1, "terminal", preview="uname -a")
            await stream.task_finished(1, "terminal", duration=1.2)
            await stream.task_started(2, "read_file", preview="foo.py")
            await stream.task_finished(2, "read_file", duration=0.3)
            await stream.stop(final_text="done")

        _run(scenario())
        tasks = [c for c in client.appended_chunks() if c.get("type") == "task_update"]
        by_id: dict[str, list[str]] = {}
        for c in tasks:
            by_id.setdefault(c["id"], []).append(c["status"])
        # One card per tool call, each transitioning in order.
        assert by_id["t1"] == ["in_progress", "complete"]
        assert by_id["t2"] == ["in_progress", "complete"]
        # Timeline order preserved: t1 opens before t2 opens.
        order = [c["id"] for c in tasks]
        assert order.index("t1") < order.index("t2")
        # stop() carries the final text + footer stats.
        stop_kwargs = [kw for name, kw in client.calls if name == "stopStream"][0]
        assert stop_kwargs.get("markdown_text") == "done"
        assert stop_kwargs.get("blocks"), "turn-stats footer missing"

    def test_failed_tool_renders_complete_with_failure_suffix(self):
        client = FakeSlackClient()
        stream = _make_stream(client)

        async def scenario():
            await stream.task_started(1, "terminal", preview="boom")
            await stream.task_finished(1, "terminal", ok=False)

        _run(scenario())
        finish = [
            c for c in client.appended_chunks()
            if c.get("id") == "t1" and c["status"] == "complete"
        ]
        assert finish and "✗ failed" in finish[0]["title"]
        # Deliberately NOT Slack's "error" status (reserved for real breakage).
        assert all(
            c["status"] != "error"
            for c in client.appended_chunks()
            if c.get("type") == "task_update"
        )


class TestRolloverReplay:
    def test_reactive_rollover_replays_in_progress_tasks(self):
        client = FakeSlackClient()
        stream = _make_stream(client)

        async def scenario():
            await stream.task_started(1, "terminal", preview="long job")
            # Next append hits the stream-expired error → reactive rollover.
            client.fail_appends_with = "message_not_in_streaming_state"
            await stream.task_started(2, "web_search", preview="q")
            # Completion for task 1 must land on the NEW stream without error.
            await stream.task_finished(1, "terminal", duration=9.0)

        _run(scenario())
        assert stream.disabled is False
        assert client.stream_opens() == 2  # original + rollover
        # The in-progress t1 card was replayed onto the fresh stream before
        # its completion arrived (otherwise Slack would drop the update).
        appended = client.appended_chunks()
        t1_updates = [c for c in appended if c.get("id") == "t1"]
        statuses = [c["status"] for c in t1_updates]
        assert statuses.count("in_progress") >= 2, (
            f"t1 not replayed after rollover — statuses: {statuses}"
        )
        assert statuses[-1] == "complete"

    def test_proactive_rollover_on_age(self):
        client = FakeSlackClient()
        stream = _make_stream(client, rollover_age_s=0.01)

        async def scenario():
            await stream.task_started(1, "terminal", preview="a")
            await asyncio.sleep(0.05)  # exceed the age threshold
            await stream.task_started(2, "read_file", preview="b")

        _run(scenario())
        assert client.stream_opens() == 2
        assert stream.disabled is False

    def test_rollover_runaway_guard_disables(self):
        client = FakeSlackClient()
        stream = _make_stream(client, rollover_age_s=0.001)
        stream.MAX_ROLLOVERS = 1

        async def scenario():
            await stream.task_started(1, "terminal", preview="a")
            await asyncio.sleep(0.01)
            await stream.task_started(2, "terminal", preview="b")  # rollover 1
            await asyncio.sleep(0.01)
            await stream.task_started(3, "terminal", preview="c")  # exceeds cap

        _run(scenario())
        assert stream.disabled is True


class TestSubagentStreams:
    def test_subagent_lifecycle_renders_numbered_card(self):
        client = FakeSlackClient()
        stream = _make_stream(client)

        async def scenario():
            await stream.subagent_event(
                "subagent.start", "abc123", goal="research X", number=1
            )
            await stream.subagent_event(
                "subagent.tool", "abc123", goal="research X", tool_name="web_search"
            )
            await stream.subagent_event(
                "subagent.complete", "abc123", goal="research X", ok=True
            )

        _run(scenario())
        cards = [c for c in client.appended_chunks() if c.get("id") == "sub_abc123"]
        assert [c["status"] for c in cards] == ["in_progress", "in_progress", "complete"]
        assert "#1" in cards[0]["title"]
        assert "🔀 Delegate" in cards[0]["title"]

    def test_parallel_subagents_get_independent_cards(self):
        client = FakeSlackClient()
        stream = _make_stream(client)

        async def scenario():
            await stream.subagent_event("subagent.start", "k1", goal="one", number=1)
            await stream.subagent_event("subagent.start", "k2", goal="two", number=2)
            await stream.subagent_event("subagent.complete", "k2", goal="two")
            await stream.subagent_event("subagent.complete", "k1", goal="one")

        _run(scenario())
        ids = {c["id"] for c in client.appended_chunks() if c.get("type") == "task_update"}
        assert {"sub_k1", "sub_k2"} <= ids

    def test_failed_subagent_marked(self):
        client = FakeSlackClient()
        stream = _make_stream(client)

        async def scenario():
            await stream.subagent_event("subagent.start", "k1", goal="g", number=1)
            await stream.subagent_event("subagent.complete", "k1", goal="g", ok=False)

        _run(scenario())
        final = [c for c in client.appended_chunks() if c.get("id") == "sub_k1"][-1]
        assert "✗ failed" in final["title"]


class TestReasoningCards:
    def test_reasoning_buffered_until_first_tool_then_flushed(self):
        """Opening thought (before any tool call) is buffered — no API calls —
        and flushed as a 💭 card by the first task_started, preserving the
        thought → tool timeline order."""
        client = FakeSlackClient()
        stream = _make_stream(client)

        async def scenario():
            await stream.reasoning_update(
                "I should inspect the repo structure before touching anything, "
                "starting with the manifest."
            )
            assert client.calls == []  # buffered, stream not opened
            await stream.task_started(1, "terminal", preview="ls")

        _run(scenario())
        chunks = client.appended_chunks()
        think = [c for c in chunks if str(c.get("id", "")).startswith("think")]
        tool = [c for c in chunks if c.get("id") == "t1"]
        assert think and tool
        # 💭 card finalized (complete) before/at the tool's open.
        assert chunks.index(think[-1]) < chunks.index(tool[0])
