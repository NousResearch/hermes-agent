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
from agent.prompt_builder import STEER_MARKER_OPEN
from tools.budget_config import BudgetConfig
from tools.tool_result_storage import PERSISTED_OUTPUT_TAG


def _assert_budget_replaced(content: str) -> None:
    """The oversized result must have been replaced by budget enforcement.

    With an active sandbox env (or host-side spillover) the replacement is a
    ``<persisted-output>`` preview+path block; when persistence is impossible
    it falls back to inline truncation. Either way the raw oversized payload
    must be gone — that is the behavior these tests pin, not which
    replacement shape was used.
    """
    assert PERSISTED_OUTPUT_TAG in content or "Truncated:" in content, content[:200]
    assert "L" * 1_000 not in content


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


    def test_never_parallel_tool_is_a_barrier(self):
        calls = [
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
            _tc("clarify", '{"question":"?"}', call_id="c1"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel", "sequential"]
        assert [tc.id for tc in segments[1][1]] == ["c1"]



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

    def test_v4a_decoy_path_does_not_parallelize_with_real_target(self, tmp_path):
        """mode=patch scopes via V4A headers, not a decoy path= argument.

        A patch that claims path=dummy.txt but updates real.py must not share
        a parallel segment with write_file/read_file on real.py.
        """
        patch_body = (
            "*** Begin Patch\n"
            "*** Update File: real.py\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )
        patch_args = json.dumps({
            "mode": "patch",
            "path": "dummy.txt",
            "patch": patch_body,
        })
        calls = [
            _tc("patch", patch_args, call_id="p1"),
            _tc("write_file", '{"path":"real.py","content":"x"}', call_id="w1"),
        ]
        segments = _plan_tool_batch_segments(calls, execution_cwd=tmp_path)
        assert _flatten_ids(segments) == ["p1", "w1"]
        # Overlap on real.py must prevent a single parallel segment.
        assert not (
            len(segments) == 1
            and segments[0][0] == "parallel"
            and [tc.id for tc in segments[0][1]] == ["p1", "w1"]
        )
        # Solo runs demote to sequential and may merge; either shape is safe.
        if len(segments) == 1:
            assert segments[0][0] == "sequential"
        else:
            assert [tc.id for tc in segments[0][1]] == ["p1"]
            assert [tc.id for tc in segments[1][1]] == ["w1"]

    def test_v4a_multi_file_reserves_all_header_targets(self, tmp_path):
        """Multi-file V4A must reserve every Update/Add/Delete/Move target."""
        patch_body = (
            "*** Begin Patch\n"
            "*** Update File: a.py\n"
            "@@\n-a\n+b\n"
            "*** Add File: b.py\n"
            "+fresh\n"
            "*** End Patch\n"
        )
        # Honest path= only names a.py — b.py still must be reserved.
        patch_args = json.dumps({
            "mode": "patch",
            "path": "a.py",
            "patch": patch_body,
        })
        calls = [
            _tc("patch", patch_args, call_id="p1"),
            _tc("read_file", '{"path":"b.py"}', call_id="r1"),
        ]
        segments = _plan_tool_batch_segments(calls, execution_cwd=tmp_path)
        assert _flatten_ids(segments) == ["p1", "r1"]
        assert not (
            len(segments) == 1
            and segments[0][0] == "parallel"
            and [tc.id for tc in segments[0][1]] == ["p1", "r1"]
        )
        if len(segments) == 1:
            assert segments[0][0] == "sequential"
        else:
            assert [tc.id for tc in segments[0][1]] == ["p1"]
            assert [tc.id for tc in segments[1][1]] == ["r1"]

    def test_v4a_without_path_arg_still_scopes_from_headers(self, tmp_path):
        """mode=patch with no path= must still parallel-scope from V4A headers."""
        patch_body = (
            "*** Begin Patch\n"
            "*** Update File: real.py\n"
            "@@\n-old\n+new\n"
            "*** End Patch\n"
        )
        patch_args = json.dumps({"mode": "patch", "patch": patch_body})
        calls = [
            _tc("patch", patch_args, call_id="p1"),
            _tc("write_file", '{"path":"other.py","content":"x"}', call_id="w1"),
            _tc("read_file", '{"path":"real.py"}', call_id="r1"),
        ]
        segments = _plan_tool_batch_segments(calls, execution_cwd=tmp_path)
        # p1+w1 are disjoint → can share a parallel run; r1 overlaps real.py → new run.
        assert _flatten_ids(segments) == ["p1", "w1", "r1"]
        assert [tc.id for tc in segments[0][1]] == ["p1", "w1"]
        assert segments[0][0] == "parallel"
        assert [tc.id for tc in segments[1][1]] == ["r1"]

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


class TestReaderWriterPathRoles:
    """Reader/writer reservation semantics on path-scoped tools.

    The originating bug: ``search_files`` was in ``_PARALLEL_SAFE_TOOLS``
    with no path reservation, so ``patch(path=X)`` + ``search_files(path=dir(X))``
    landed in ONE parallel segment and the search could observe pre-patch
    file content (stale-read race).  Fix: ``search_files`` reserves its
    search root as a READER; overlap conflicts only when a WRITER is on
    either side.
    """

    def test_search_files_after_patch_same_subtree_splits(self, tmp_path, monkeypatch):
        """The exact smoke-test race: patch a file, search its directory."""
        monkeypatch.chdir(tmp_path)
        calls = [
            _tc("patch", '{"path":"scratch/sample.txt","old_string":"a","new_string":"patched"}', call_id="w1"),
            _tc("search_files", '{"pattern":"patched","path":"scratch"}', call_id="s1"),
        ]
        segments = _plan_tool_batch_segments(calls, execution_cwd=tmp_path)
        # Both calls survive, but never in the same PARALLEL segment.
        # (A shared *sequential* segment is fine — sequential is ordered.)
        assert _flatten_ids(segments) == ["w1", "s1"]
        for kind, seg_calls in segments:
            ids = [tc.id for tc in seg_calls]
            assert not (kind == "parallel" and {"w1", "s1"} <= set(ids)), (
                "write and dependent search must not share a parallel segment"
            )

    def test_search_files_default_root_conflicts_with_write_into_cwd(self, tmp_path, monkeypatch):
        """search_files with NO path arg reserves the cwd — a write anywhere
        under the cwd must not share its segment."""
        monkeypatch.chdir(tmp_path)
        calls = [
            _tc("write_file", '{"path":"out/notes.txt","content":"x"}', call_id="w1"),
            _tc("search_files", '{"pattern":"notes"}', call_id="s1"),
        ]
        segments = _plan_tool_batch_segments(calls, execution_cwd=tmp_path)
        for kind, seg_calls in segments:
            ids = [tc.id for tc in seg_calls]
            assert not (kind == "parallel" and {"w1", "s1"} <= set(ids))

    def test_reader_reader_same_file_stays_parallel(self, tmp_path, monkeypatch):
        """Two reads of the same file commute — the old planner needlessly
        split them; they must now share one parallel segment."""
        monkeypatch.chdir(tmp_path)
        calls = [
            _tc("read_file", '{"path":"a.py"}', call_id="r1"),
            _tc("read_file", '{"path":"a.py"}', call_id="r2"),
        ]
        segments = _plan_tool_batch_segments(calls, execution_cwd=tmp_path)
        assert _kinds(segments) == ["parallel"]
        assert [tc.id for tc in segments[0][1]] == ["r1", "r2"]

    def test_read_file_and_search_files_overlapping_stay_parallel(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = [
            _tc("read_file", '{"path":"src/a.py"}', call_id="r1"),
            _tc("search_files", '{"pattern":"foo","path":"src"}', call_id="s1"),
        ]
        segments = _plan_tool_batch_segments(calls, execution_cwd=tmp_path)
        assert _kinds(segments) == ["parallel"]

    def test_search_files_disjoint_from_write_stays_parallel(self, tmp_path, monkeypatch):
        """A search rooted outside the written subtree has no conflict."""
        monkeypatch.chdir(tmp_path)
        calls = [
            _tc("write_file", '{"path":"src/a.py","content":"x"}', call_id="w1"),
            _tc("search_files", '{"pattern":"foo","path":"docs"}', call_id="s1"),
        ]
        segments = _plan_tool_batch_segments(calls, execution_cwd=tmp_path)
        assert _kinds(segments) == ["parallel"]
        assert [tc.id for tc in segments[0][1]] == ["w1", "s1"]

    def test_writer_writer_same_path_still_splits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = [
            _tc("write_file", '{"path":"a.py","content":"1"}', call_id="w1"),
            _tc("write_file", '{"path":"a.py","content":"2"}', call_id="w2"),
        ]
        segments = _plan_tool_batch_segments(calls, execution_cwd=tmp_path)
        for kind, seg_calls in segments:
            ids = [tc.id for tc in seg_calls]
            assert not (kind == "parallel" and {"w1", "w2"} <= set(ids))

    def test_read_then_write_same_file_still_splits(self, tmp_path, monkeypatch):
        """Reader followed by writer on the same path keeps the pre-existing
        split (write must not clobber a file mid-read)."""
        monkeypatch.chdir(tmp_path)
        calls = [
            _tc("read_file", '{"path":"a.py"}', call_id="r1"),
            _tc("write_file", '{"path":"a.py","content":"x"}', call_id="w1"),
        ]
        segments = _plan_tool_batch_segments(calls, execution_cwd=tmp_path)
        for kind, seg_calls in segments:
            ids = [tc.id for tc in seg_calls]
            assert not (kind == "parallel" and {"r1", "w1"} <= set(ids))


class TestShouldParallelizeBackwardCompat:
    """The boolean gate is now a view over the planner — same answers as before."""

    def test_single_call_is_sequential(self):
        assert not _should_parallelize_tool_batch([_tc("web_search")])





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
            "model_tools.get_tool_definitions",
            return_value=_make_tool_defs("web_search", "terminal"),
        ),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
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

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
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

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls(msg, messages, "task-1")

        # Every call still gets exactly one result, in order.
        assert [m["tool_call_id"] for m in messages] == ["s1", "s2", "t1", "s3", "s4"]
        # s3/s4 were never executed.
        assert "s3" not in executed and "s4" not in executed
        for m in messages[-2:]:
            assert "cancelled" in m["content"] or "skipped" in m["content"]

    def test_steer_lands_exactly_once_in_mixed_batch(self, agent):
        """The whole-batch finalizer drains steer once, so the marker cannot
        be duplicated by segment boundaries."""
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
        with patch("model_tools.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls(msg, messages, "task-1")

        contents = [m["content"] for m in messages]
        hits = [c for c in contents if "focus on the tests" in c]
        assert len(hits) == 1

    @pytest.mark.parametrize(
        ("calls", "expected_segment_kinds"),
        [
            (
                [
                    _tc("web_search", '{"query":"large"}', call_id="parallel-large"),
                    _tc("web_search", '{"query":"small"}', call_id="parallel-small"),
                ],
                ["parallel"],
            ),
            (
                [
                    _tc("terminal", '{"command":"large"}', call_id="sequential-large"),
                    _tc("terminal", '{"command":"small"}', call_id="sequential-small"),
                ],
                ["sequential"],
            ),
            (
                [
                    _tc("web_search", '{"query":"large"}', call_id="mixed-large"),
                    _tc("web_search", '{"query":"small"}', call_id="mixed-search-small"),
                    _tc("terminal", '{"command":"small"}', call_id="mixed-terminal-small"),
                ],
                ["parallel", "sequential"],
            ),
            (
                [
                    _tc("web_search", '{"query":"small"}', call_id="mixed-search-first-small"),
                    _tc("web_search", '{"query":"small"}', call_id="mixed-search-second-small"),
                    _tc("terminal", '{"command":"large"}', call_id="mixed-terminal-large"),
                ],
                ["parallel", "sequential"],
            ),
        ],
        ids=["parallel", "sequential", "mixed-parallel-large", "mixed-sequential-large"],
    )
    def test_steer_survives_turn_budget_in_every_dispatch_path(
        self, agent, calls, expected_segment_kinds
    ):
        """A steer must be appended after aggregate budgeting in direct
        concurrent, direct sequential, and segmented mixed batches.

        The large result forces ``enforce_turn_budget()`` to replace it.
        Before the fix, the per-tool drain consumed the steer first, so that
        replacement silently discarded the canonical marker.
        """
        messages = []
        msg = SimpleNamespace(content="", tool_calls=calls)
        budget = BudgetConfig(
            default_result_size=10_000,
            turn_budget=48,
            preview_size=16,
        )

        assert _kinds(_plan_tool_batch_segments(calls)) == expected_segment_kinds

        def fake_handle(name, args, task_id, **kwargs):
            if kwargs["tool_call_id"].endswith("large"):
                assert agent.steer("preserve this steer after budget enforcement")
                return "L" * 1_000
            return "small"

        with (
            patch("model_tools.handle_function_call", side_effect=fake_handle),
            patch("agent.tool_executor._budget_for_agent", return_value=budget),
        ):
            agent._execute_tool_calls(msg, messages, "task-1")

        large_result_index = next(i for i, call in enumerate(calls) if call.id.endswith("large"))
        _assert_budget_replaced(messages[large_result_index]["content"])
        steer_messages = [m for m in messages if STEER_MARKER_OPEN in m["content"]]
        assert steer_messages == [messages[-1]]
        assert "preserve this steer after budget enforcement" in steer_messages[0]["content"]

    def test_steer_survives_turn_budget_after_malformed_arguments(self, agent):
        """Malformed arguments still reach the shared post-budget finalizer.

        The parser error itself can exceed a constrained turn budget.  A steer
        queued before that malformed sequential call must therefore remain
        pending until after the error result is replaced by the budget preview.
        """
        calls = [_tc("terminal", "{not json", call_id="malformed")]
        messages = []
        msg = SimpleNamespace(content="", tool_calls=calls)
        budget = BudgetConfig(
            default_result_size=10_000,
            turn_budget=48,
            preview_size=16,
        )

        assert _kinds(_plan_tool_batch_segments(calls)) == ["sequential"]
        assert agent.steer("preserve malformed-call steer after budget enforcement")

        with patch("agent.tool_executor._budget_for_agent", return_value=budget):
            agent._execute_tool_calls(msg, messages, "task-1")

        assert len(messages) == 1
        _assert_budget_replaced(messages[0]["content"])
        assert messages[0]["content"].count(STEER_MARKER_OPEN) == 1
        assert "preserve malformed-call steer after budget enforcement" in messages[0]["content"]


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


    # ``windows_only`` rather than ``skipif(sys.platform != "win32")``: the
    # Windows CI job greps for the marker to decide which files to import, so
    # a bare skipif leaves this running on no host at all.
    @pytest.mark.windows_only
    def test_case_insensitive_paths_overlap_windows(self, tmp_path):
        """On Windows, FILE.txt and file.txt are the same file — they must
        be detected as overlapping after normcase() canonicalisation."""
        from agent.tool_dispatch_helpers import _canonical_path, _paths_overlap

        upper = _canonical_path(str(tmp_path / "FILE.txt"), execution_cwd=tmp_path)
        lower = _canonical_path(str(tmp_path / "file.txt"), execution_cwd=tmp_path)

        assert _paths_overlap(upper, lower), (
            "Case-insensitive aliases must overlap on Windows"
        )
