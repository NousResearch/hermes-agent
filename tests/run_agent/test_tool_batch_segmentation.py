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
import sys
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


class TestPathCanonicalization:
    """Regression tests for _canonical_path / _extract_parallel_scope_path fixes.

    Verifies that symlink aliases, relative/absolute cwd mismatches, and
    (on Windows) case-insensitive aliases are never placed in the same
    parallel segment.
    """

    def test_relative_and_absolute_same_target_use_separate_segments(self, tmp_path):
        """A relative path resolved against execution_cwd and an absolute path
        pointing to the same file must be detected as overlapping."""
        from agent.tool_dispatch_helpers import (
            _canonical_path,
            _paths_overlap,
        )

        target = tmp_path / "config.json"
        target.touch()

        abs_path = _canonical_path(str(target))
        rel_path = _canonical_path("config.json", execution_cwd=tmp_path)

        assert _paths_overlap(abs_path, rel_path), (
            "Absolute and relative paths pointing to the same file must overlap"
        )

    def test_symlink_aliases_are_not_parallelized(self, tmp_path):
        """A symlink alias and the real path must be detected as overlapping
        so they are never placed in the same parallel segment."""
        import os
        from agent.tool_dispatch_helpers import (
            _canonical_path,
            _paths_overlap,
        )

        real_dir = tmp_path / "real"
        real_dir.mkdir()
        target = real_dir / "config.json"
        target.touch()

        alias_dir = tmp_path / "alias"
        alias_dir.symlink_to(real_dir)

        real_path = _canonical_path(str(target))
        alias_path = _canonical_path(str(alias_dir / "config.json"))

        assert _paths_overlap(real_path, alias_path), (
            "Symlink alias and real path must overlap — "
            "they must not be parallelized"
        )

    def test_execution_cwd_used_over_process_cwd(self, tmp_path, monkeypatch):
        """_extract_parallel_scope_path must use execution_cwd, not
        process cwd, when resolving relative paths."""
        from agent.tool_dispatch_helpers import (
            _extract_parallel_scope_path,
            _paths_overlap,
        )

        exec_cwd = tmp_path / "sub"
        exec_cwd.mkdir()
        (exec_cwd / "x.txt").touch()

        # Point process cwd somewhere else entirely.
        monkeypatch.chdir(tmp_path)

        # With execution_cwd supplied the relative path resolves under exec_cwd.
        path_with_cwd = _extract_parallel_scope_path(
            "write_file", {"path": "x.txt"}, execution_cwd=exec_cwd
        )
        # The absolute path under exec_cwd must match.
        path_absolute = _extract_parallel_scope_path(
            "write_file", {"path": str(exec_cwd / "x.txt")}
        )

        assert path_with_cwd is not None
        assert path_absolute is not None
        assert _paths_overlap(path_with_cwd, path_absolute), (
            "execution_cwd-relative path and absolute path must overlap; "
            "process cwd must not be used when execution_cwd is provided"
        )

    def test_symlink_alias_nonexistent_write_target_overlap(self, tmp_path):
        """Symlink parent + not-yet-created leaf file must still be detected
        as overlapping — write_file targets may not exist at planning time."""
        import os
        from agent.tool_dispatch_helpers import _canonical_path, _paths_overlap

        real_dir = tmp_path / "real"
        real_dir.mkdir()
        alias_dir = tmp_path / "alias"
        alias_dir.symlink_to(real_dir)

        # Leaf file does NOT exist yet (write_file scenario).
        real_target = _canonical_path(str(real_dir / "new.txt"))
        alias_target = _canonical_path(str(alias_dir / "new.txt"))

        assert _paths_overlap(real_target, alias_target), (
            "Symlink parent + nonexistent leaf must overlap — "
            "write_file targets are planned before they exist"
        )

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="normcase() case-folding only matters on Windows",
    )
    def test_case_insensitive_paths_overlap_windows(self, tmp_path):
        """On Windows, FILE.txt and file.txt are the same file — they must
        be detected as overlapping after normcase() canonicalisation."""
        from agent.tool_dispatch_helpers import _canonical_path, _paths_overlap

        upper = _canonical_path(str(tmp_path / "FILE.txt"), execution_cwd=tmp_path)
        lower = _canonical_path(str(tmp_path / "file.txt"), execution_cwd=tmp_path)

        assert _paths_overlap(upper, lower), (
            "Case-insensitive aliases must overlap on Windows"
        )
