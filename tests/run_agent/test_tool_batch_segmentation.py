"""Segment-aware mixed tool-batch dispatch.

A model response containing several parallel-safe reads plus one unsafe
tool used to lose ALL concurrency: `_should_parallelize_tool_batch` was
all-or-nothing, so one barrier call forced the entire batch onto the
sequential path.  `_plan_tool_batch_segments` now splits the batch into
ordered segments — maximal contiguous runs of parallel-safe calls execute
concurrently, barrier calls sequentially — while preserving:

  * model tool-result ordering (one result per call, in emission order),
  * side-effect boundaries (no call starts before an earlier barrier ends).
"""

import json
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.tool_dispatch_helpers import (
    _plan_tool_batch_segments,
    _should_parallelize_tool_batch,
)


def _tc(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _kinds(segments):
    return [kind for kind, _ in segments]


def _flatten_ids(segments):
    return [tc.id for _, calls in segments for tc in calls]


# ---------------------------------------------------------------------------
# Planner unit tests
# ---------------------------------------------------------------------------


class TestPlanToolBatchSegments:
    def test_all_safe_batch_is_single_parallel_segment(self):
        calls = [_tc("web_search"), _tc("read_file", '{"path":"a.py"}'), _tc("web_extract")]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel"]
        assert _flatten_ids(segments) == [c.id for c in calls]

    def test_three_safe_reads_plus_trailing_unsafe_keeps_reads_parallel(self):
        """The headline case: 3 safe reads + 1 unsafe tool must NOT go fully sequential."""
        calls = [
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
            _tc("read_file", '{"path":"a.py"}', call_id="r3"),
            _tc("terminal", '{"command":"echo hi"}', call_id="b1"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel", "sequential"]
        assert [tc.id for tc in segments[0][1]] == ["r1", "r2", "r3"]
        assert [tc.id for tc in segments[1][1]] == ["b1"]

    def test_barrier_in_middle_splits_runs_and_preserves_order(self):
        calls = [
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
            _tc("terminal", '{"command":"make"}', call_id="b1"),
            _tc("web_search", call_id="r3"),
            _tc("web_search", call_id="r4"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel", "sequential", "parallel"]
        assert _flatten_ids(segments) == ["r1", "r2", "b1", "r3", "r4"]

    def test_single_safe_call_after_barrier_is_demoted_and_merged(self):
        # parallel run of 1 gains nothing — demote to sequential and merge
        # with the adjacent barrier segment.
        calls = [
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
            _tc("terminal", '{"command":"make"}', call_id="b1"),
            _tc("web_search", call_id="r3"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel", "sequential"]
        assert [tc.id for tc in segments[1][1]] == ["b1", "r3"]

    def test_adjacent_barriers_merge_into_one_sequential_segment(self):
        calls = [
            _tc("terminal", '{"command":"a"}', call_id="b1"),
            _tc("terminal", '{"command":"b"}', call_id="b2"),
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["sequential", "parallel"]
        assert [tc.id for tc in segments[0][1]] == ["b1", "b2"]

    def test_never_parallel_tool_is_a_barrier(self):
        calls = [
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
            _tc("clarify", '{"question":"?"}', call_id="c1"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel", "sequential"]
        assert [tc.id for tc in segments[1][1]] == ["c1"]

    def test_malformed_args_call_is_a_barrier_not_a_batch_poison(self):
        calls = [
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
            _tc("web_search", "{not json", call_id="bad"),
            _tc("web_search", call_id="r3"),
            _tc("web_search", call_id="r4"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel", "sequential", "parallel"]
        assert [tc.id for tc in segments[1][1]] == ["bad"]

    def test_non_dict_args_call_is_a_barrier(self):
        calls = [
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
            _tc("web_search", '"just a string"', call_id="bad"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel", "sequential"]

    def test_overlapping_paths_split_across_segments(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = [
            _tc("read_file", '{"path":"a.py"}', call_id="w1"),
            _tc("web_search", call_id="r1"),
            _tc("write_file", '{"path":"a.py","content":"x"}', call_id="w2"),
            _tc("web_search", call_id="r2"),
        ]
        segments = _plan_tool_batch_segments(calls)
        # w2 conflicts with w1 → closes the first run; w2+r2 form the second.
        assert _kinds(segments) == ["parallel", "parallel"]
        assert [tc.id for tc in segments[0][1]] == ["w1", "r1"]
        assert [tc.id for tc in segments[1][1]] == ["w2", "r2"]
        # Order and completeness preserved.
        assert _flatten_ids(segments) == ["w1", "r1", "w2", "r2"]

    def test_path_scoped_tool_without_path_is_a_barrier(self):
        calls = [
            _tc("read_file", "{}", call_id="nopath"),
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["sequential", "parallel"]

    def test_flattened_segments_always_preserve_emission_order(self):
        calls = [
            _tc("terminal", '{"command":"x"}', call_id="b1"),
            _tc("web_search", call_id="r1"),
            _tc("clarify", '{"question":"?"}', call_id="c1"),
            _tc("read_file", '{"path":"a.py"}', call_id="r2"),
            _tc("read_file", '{"path":"b.py"}', call_id="r3"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _flatten_ids(segments) == ["b1", "r1", "c1", "r2", "r3"]


class TestShouldParallelizeBackwardCompat:
    """The boolean gate is now a view over the planner — same answers as before."""

    def test_single_call_is_sequential(self):
        assert not _should_parallelize_tool_batch([_tc("web_search")])

    def test_all_safe_batch_is_parallel(self):
        assert _should_parallelize_tool_batch([_tc("web_search"), _tc("web_extract")])

    def test_mixed_batch_is_not_wholly_parallel(self):
        assert not _should_parallelize_tool_batch(
            [_tc("web_search"), _tc("terminal", '{"command":"ls"}')]
        )

    def test_clarify_anywhere_blocks_whole_batch_parallelism(self):
        assert not _should_parallelize_tool_batch(
            [_tc("web_search"), _tc("clarify", '{"question":"?"}')]
        )


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search", "terminal"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


class TestSegmentedDispatchIntegration:
    def test_mixed_batch_runs_safe_prefix_concurrently_and_barrier_after(self, agent):
        """Two web_search calls must overlap in time; terminal must start only
        after both finish; results land in the model's emission order."""
        calls = [
            _tc("web_search", '{"query":"a"}', call_id="s1"),
            _tc("web_search", '{"query":"b"}', call_id="s2"),
            _tc("terminal", '{"command":"echo done"}', call_id="t1"),
        ]
        msg = SimpleNamespace(content="", tool_calls=calls)
        messages = []

        rendezvous = threading.Barrier(2, timeout=10)
        events = []
        events_lock = threading.Lock()

        def fake_handle(name, args, task_id, **kwargs):
            with events_lock:
                events.append(("start", name, kwargs["tool_call_id"]))
            if name == "web_search":
                # Both searches must be in flight at once to pass this
                # barrier — proves genuine concurrency for the safe prefix.
                rendezvous.wait()
            with events_lock:
                events.append(("end", name, kwargs["tool_call_id"]))
            return json.dumps({"ok": name})

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls(msg, messages, "task-1")

        # One result per call, in emission order.
        assert [m["tool_call_id"] for m in messages] == ["s1", "s2", "t1"]
        assert all(m["role"] == "tool" for m in messages)

        # The barrier (terminal) started only after BOTH searches ended.
        terminal_start = events.index(("start", "terminal", "t1"))
        search_ends = [
            i for i, e in enumerate(events) if e[0] == "end" and e[1] == "web_search"
        ]
        assert len(search_ends) == 2
        assert all(i < terminal_start for i in search_ends)

    def test_mixed_batch_preserves_order_with_barrier_in_middle(self, agent):
        calls = [
            _tc("web_search", '{"query":"a"}', call_id="s1"),
            _tc("web_search", '{"query":"b"}', call_id="s2"),
            _tc("terminal", '{"command":"touch x"}', call_id="t1"),
            _tc("web_search", '{"query":"c"}', call_id="s3"),
            _tc("web_search", '{"query":"d"}', call_id="s4"),
        ]
        msg = SimpleNamespace(content="", tool_calls=calls)
        messages = []
        executed = []
        lock = threading.Lock()

        def fake_handle(name, args, task_id, **kwargs):
            with lock:
                executed.append(kwargs["tool_call_id"])
            return json.dumps({"ok": True})

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls(msg, messages, "task-1")

        assert [m["tool_call_id"] for m in messages] == ["s1", "s2", "t1", "s3", "s4"]
        # Barrier ordering: t1 executed after {s1,s2} and before {s3,s4}.
        t1_pos = executed.index("t1")
        assert {"s1", "s2"} == set(executed[:t1_pos])
        assert {"s3", "s4"} == set(executed[t1_pos + 1:])

    def test_homogeneous_safe_batch_still_uses_plain_concurrent_path(self, agent):
        calls = [_tc("web_search", '{"query":"a"}'), _tc("web_search", '{"query":"b"}')]
        msg = SimpleNamespace(content="", tool_calls=calls)

        with (
            patch.object(agent, "_execute_tool_calls_concurrent") as conc,
            patch.object(agent, "_execute_tool_calls_sequential") as seq,
        ):
            agent._execute_tool_calls(msg, [], "task-1")

        conc.assert_called_once()
        seq.assert_not_called()

    def test_homogeneous_unsafe_batch_still_uses_plain_sequential_path(self, agent):
        calls = [
            _tc("terminal", '{"command":"a"}'),
            _tc("terminal", '{"command":"b"}'),
        ]
        msg = SimpleNamespace(content="", tool_calls=calls)

        with (
            patch.object(agent, "_execute_tool_calls_concurrent") as conc,
            patch.object(agent, "_execute_tool_calls_sequential") as seq,
        ):
            agent._execute_tool_calls(msg, [], "task-1")

        seq.assert_called_once()
        conc.assert_not_called()

    def test_single_call_uses_sequential_path(self, agent):
        msg = SimpleNamespace(content="", tool_calls=[_tc("web_search", '{"query":"a"}')])

        with (
            patch.object(agent, "_execute_tool_calls_concurrent") as conc,
            patch.object(agent, "_execute_tool_calls_sequential") as seq,
        ):
            agent._execute_tool_calls(msg, [], "task-1")

        seq.assert_called_once()
        conc.assert_not_called()

    def test_interrupt_during_barrier_drains_later_segments(self, agent):
        """Interrupt raised while the barrier tool runs: the trailing parallel
        segment must be drained with cancelled results — one per call —
        without executing."""
        calls = [
            _tc("web_search", '{"query":"a"}', call_id="s1"),
            _tc("web_search", '{"query":"b"}', call_id="s2"),
            _tc("terminal", '{"command":"long"}', call_id="t1"),
            _tc("web_search", '{"query":"c"}', call_id="s3"),
            _tc("web_search", '{"query":"d"}', call_id="s4"),
        ]
        msg = SimpleNamespace(content="", tool_calls=calls)
        messages = []
        executed = []
        lock = threading.Lock()

        def fake_handle(name, args, task_id, **kwargs):
            with lock:
                executed.append(kwargs["tool_call_id"])
            if kwargs["tool_call_id"] == "t1":
                agent._interrupt_requested = True
            return json.dumps({"ok": True})

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls(msg, messages, "task-1")

        # Every call still gets exactly one result, in order.
        assert [m["tool_call_id"] for m in messages] == ["s1", "s2", "t1", "s3", "s4"]
        # s3/s4 were never executed.
        assert "s3" not in executed and "s4" not in executed
        for m in messages[-2:]:
            assert "cancelled" in m["content"] or "skipped" in m["content"]

    def test_steer_lands_exactly_once_in_mixed_batch(self, agent):
        """Steer is drained once (per-tool drains + one dispatcher-level
        finalize) — the marker must appear exactly once across the batch,
        never duplicated by segment boundaries."""
        calls = [
            _tc("web_search", '{"query":"a"}', call_id="s1"),
            _tc("web_search", '{"query":"b"}', call_id="s2"),
            _tc("terminal", '{"command":"echo hi"}', call_id="t1"),
        ]
        msg = SimpleNamespace(content="", tool_calls=calls)
        messages = []

        def fake_handle(name, args, task_id, **kwargs):
            return json.dumps({"ok": True})

        agent.steer("focus on the tests")
        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls(msg, messages, "task-1")

        contents = [m["content"] for m in messages]
        hits = [c for c in contents if "focus on the tests" in c]
        assert len(hits) == 1


class TestDanglingMutatingWorkerGuard:
    """Regression tests for the timed-out mutating worker guard in
    execute_tool_calls_segmented.

    When a parallel segment contains a write_file or patch call that times out,
    the worker thread is left running detached. A later segment that targets
    the same file must be blocked to prevent the detached worker from
    overwriting the later result.
    """

    def _make_agent(self):
        from unittest.mock import MagicMock
        agent = MagicMock()
        agent._interrupt_requested = False
        agent._tool_worker_threads_lock = __import__("threading").Lock()
        agent._tool_worker_threads = []
        agent.tool_delay = 0
        agent.log_prefix = ""
        return agent

    def test_later_mutation_blocked_after_timeout(self):
        """If write_file times out in segment 1, a later write_file in
        segment 2 must be blocked with an error result, not executed."""
        import json
        from unittest.mock import patch, MagicMock
        from types import SimpleNamespace
        from agent.tool_executor import execute_tool_calls_segmented

        # Simulate segment 1: write_file timed out (result contains "timed out")
        # Simulate segment 2: another write_file
        timeout_msg = {"role": "tool", "tool_call_id": "w1", "content": "Error executing tool 'write_file': timed out after 30.0s __concurrent_tool_timed_out__"}
        tc1 = SimpleNamespace(function=SimpleNamespace(name="write_file", arguments='{"path":"a.py","content":"x"}'), id="w1")
        tc2 = SimpleNamespace(function=SimpleNamespace(name="write_file", arguments='{"path":"a.py","content":"y"}'), id="w2")

        segments = [
            ("parallel", [tc1]),
            ("sequential", [tc2]),
        ]

        messages = []
        agent = self._make_agent()

        def fake_concurrent(agent, msg, messages, *args, **kwargs):
            # Simulate write_file timing out
            messages.append(timeout_msg)

        def fake_sequential(agent, msg, messages, *args, **kwargs):
            # Should NOT be called for tc2 — it must be blocked
            messages.append({"role": "tool", "tool_call_id": "w2", "content": '{"bytes_written": 10}'})

        with patch("agent.tool_executor.execute_tool_calls_concurrent", side_effect=fake_concurrent), \
             patch("agent.tool_executor.execute_tool_calls_sequential", side_effect=fake_sequential), \
             patch("agent.tool_executor._plan_tool_batch_segments"), \
             patch("agent.tool_executor.enforce_turn_budget"), \
             patch("agent.tool_executor._budget_for_agent", return_value=None), \
             patch("agent.tool_executor.get_active_env", return_value=None):
            assistant_message = SimpleNamespace(tool_calls=[tc1, tc2])
            execute_tool_calls_segmented(
                agent, assistant_message, messages, "task-1", segments=segments
            )

        # w1 result: timeout
        assert messages[0]["tool_call_id"] == "w1"
        assert "timed out" in messages[0]["content"]

        # w2 result: must be blocked, not the successful write result
        assert messages[1]["tool_call_id"] == "w2"
        result = json.loads(messages[1]["content"])
        assert "error" in result
        assert "blocked" in result["error"] or "dangling" in result["error"] or "timed-out" in result["error"]

    def test_read_only_segment_not_blocked_after_timeout(self):
        """If write_file times out in segment 1, a later read-only segment
        must NOT be blocked — only file mutations are guarded."""
        import json
        from unittest.mock import patch, MagicMock
        from types import SimpleNamespace
        from agent.tool_executor import execute_tool_calls_segmented

        timeout_msg = {"role": "tool", "tool_call_id": "w1", "content": "Error executing tool 'write_file': timed out after 30.0s __concurrent_tool_timed_out__"}
        read_result = {"role": "tool", "tool_call_id": "r1", "content": '{"content": "hello"}'}

        tc1 = SimpleNamespace(function=SimpleNamespace(name="write_file", arguments='{"path":"a.py","content":"x"}'), id="w1")
        tc2 = SimpleNamespace(function=SimpleNamespace(name="read_file", arguments='{"path":"a.py"}'), id="r1")

        segments = [
            ("parallel", [tc1]),
            ("sequential", [tc2]),
        ]

        messages = []
        agent = self._make_agent()

        def fake_concurrent(agent, msg, messages, *args, **kwargs):
            messages.append(timeout_msg)

        def fake_sequential(agent, msg, messages, *args, **kwargs):
            messages.append(read_result)

        with patch("agent.tool_executor.execute_tool_calls_concurrent", side_effect=fake_concurrent), \
             patch("agent.tool_executor.execute_tool_calls_sequential", side_effect=fake_sequential), \
             patch("agent.tool_executor._plan_tool_batch_segments"), \
             patch("agent.tool_executor.enforce_turn_budget"), \
             patch("agent.tool_executor._budget_for_agent", return_value=None), \
             patch("agent.tool_executor.get_active_env", return_value=None):
            assistant_message = SimpleNamespace(tool_calls=[tc1, tc2])
            execute_tool_calls_segmented(
                agent, assistant_message, messages, "task-1", segments=segments
            )

        # r1 must have executed normally, not been blocked
        assert messages[1]["tool_call_id"] == "r1"
        assert "content" in json.loads(messages[1]["content"])
