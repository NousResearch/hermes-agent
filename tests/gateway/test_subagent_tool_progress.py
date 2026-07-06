"""Tests for the subagent_tool_progress display flag and its gateway formatters.

Covers:
  * gateway.display_config resolution + tri-state normalisation
  * gateway.run._format_subagent_tool_card  ("full" mode, one child tool call)
  * gateway.run._format_subagent_progress_card ("batched" mode summary)

The flag controls whether a delegate_task child's OWN tool calls surface in
gateway progress. Child tool events are already relayed from
tools/delegate_tool.py as "subagent.tool" / "subagent.progress"; the gateway
drops them unless this flag opts in.
"""


# ---------------------------------------------------------------------------
# display_config: resolution + normalisation
# ---------------------------------------------------------------------------

class TestSubagentToolProgressResolution:
    def test_default_is_off_everywhere(self):
        from gateway.display_config import resolve_display_setting

        # No config anywhere → global default "off".
        assert resolve_display_setting({}, "telegram", "subagent_tool_progress") == "off"
        assert resolve_display_setting({}, "discord", "subagent_tool_progress") == "off"
        assert resolve_display_setting({}, "unknown", "subagent_tool_progress") == "off"

    def test_global_setting_applies(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"subagent_tool_progress": "batched"}}
        assert (
            resolve_display_setting(config, "telegram", "subagent_tool_progress")
            == "batched"
        )

    def test_platform_override_wins(self):
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "subagent_tool_progress": "batched",
                "platforms": {"telegram": {"subagent_tool_progress": "full"}},
            }
        }
        assert (
            resolve_display_setting(config, "telegram", "subagent_tool_progress")
            == "full"
        )
        # Other platforms still see the global value.
        assert (
            resolve_display_setting(config, "discord", "subagent_tool_progress")
            == "batched"
        )

    def test_normalise_unknown_string_fails_safe_to_off(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"platforms": {"telegram": {"subagent_tool_progress": "loud"}}}}
        assert (
            resolve_display_setting(config, "telegram", "subagent_tool_progress")
            == "off"
        )

    def test_normalise_legacy_booleans(self):
        from gateway.display_config import resolve_display_setting

        true_cfg = {"display": {"platforms": {"telegram": {"subagent_tool_progress": True}}}}
        false_cfg = {"display": {"platforms": {"telegram": {"subagent_tool_progress": False}}}}
        assert (
            resolve_display_setting(true_cfg, "telegram", "subagent_tool_progress")
            == "full"
        )
        # NOTE: bare `off`/`False` in YAML 1.1 both arrive as Python False here.
        assert (
            resolve_display_setting(false_cfg, "telegram", "subagent_tool_progress")
            == "off"
        )

    def test_normalise_is_case_insensitive(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"platforms": {"telegram": {"subagent_tool_progress": "FULL"}}}}
        assert (
            resolve_display_setting(config, "telegram", "subagent_tool_progress")
            == "full"
        )

    def test_flag_is_in_overrideable_keys(self):
        from gateway.display_config import OVERRIDEABLE_KEYS

        assert "subagent_tool_progress" in OVERRIDEABLE_KEYS


# ---------------------------------------------------------------------------
# run._format_subagent_tool_card — "full" mode
# ---------------------------------------------------------------------------

class TestFormatSubagentToolCard:
    def test_single_subagent_no_index_tag(self):
        from gateway.run import _format_subagent_tool_card

        card = _format_subagent_tool_card(
            "terminal",
            "pytest -q",
            goal="Run the test suite",
            task_index=0,
            task_count=1,
        )
        # Header carries the goal, no "[1]" tag for a single subagent.
        assert "Run the test suite" in card
        assert "[1]" not in card
        # Tool line present with its preview.
        assert "terminal" in card
        assert "pytest -q" in card

    def test_batch_subagent_shows_index_tag(self):
        from gateway.run import _format_subagent_tool_card

        card = _format_subagent_tool_card(
            "read_file",
            "config.yaml",
            goal="Audit config",
            task_index=2,
            task_count=4,
        )
        # 1-indexed tag for parallel batches.
        assert "[3]" in card
        assert "read_file" in card

    def test_long_preview_is_truncated(self):
        from gateway.run import _format_subagent_tool_card

        long_preview = "x" * 200
        card = _format_subagent_tool_card(
            "search_files",
            long_preview,
            goal="g",
            task_index=0,
            task_count=1,
        )
        # Capped well under the raw length (default cap 40).
        assert "x" * 200 not in card
        assert "…" in card

    def test_missing_goal_falls_back_to_subagent_label(self):
        from gateway.run import _format_subagent_tool_card

        card = _format_subagent_tool_card("web_search", "weather", goal=None)
        assert "subagent" in card
        assert "web_search" in card

    def test_never_includes_output_marker(self):
        from gateway.run import _format_subagent_tool_card

        # The preview is tool INPUT only; the formatter should not invent any
        # output/summary framing.
        card = _format_subagent_tool_card("terminal", "ls", goal="list")
        assert "summary" not in card.lower()
        assert "output" not in card.lower()

    def test_include_header_false_omits_goal_header(self):
        from gateway.run import _format_subagent_tool_card

        card = _format_subagent_tool_card(
            "read_file",
            "config.yaml",
            goal="Audit config",
            include_header=False,
        )
        # No "🔀" header line; goal not repeated; single indented tool line.
        assert "🔀" not in card
        assert "Audit config" not in card
        assert "\n" not in card  # exactly one line
        assert card.startswith("└ ")
        assert "read_file" in card and "config.yaml" in card

    def test_include_header_true_is_default(self):
        from gateway.run import _format_subagent_tool_card

        with_default = _format_subagent_tool_card("terminal", "ls", goal="g")
        explicit = _format_subagent_tool_card(
            "terminal", "ls", goal="g", include_header=True
        )
        assert with_default == explicit
        assert "🔀" in with_default
        assert with_default.count("\n") == 1  # header + tool line


# ---------------------------------------------------------------------------
# run._format_subagent_progress_card — "batched" mode
# ---------------------------------------------------------------------------

class TestFormatSubagentProgressCard:
    def test_passthrough_of_batched_summary(self):
        from gateway.run import _format_subagent_progress_card

        summary = "🔀 terminal, read_file, search_files, terminal, web_search"
        assert _format_subagent_progress_card(summary) == summary

    def test_empty_preview_returns_none(self):
        from gateway.run import _format_subagent_progress_card

        assert _format_subagent_progress_card("") is None
        assert _format_subagent_progress_card(None) is None
        assert _format_subagent_progress_card("   ") is None


# ---------------------------------------------------------------------------
# progress_callback dispatch — exercises the real branch via a faithful harness
# ---------------------------------------------------------------------------
# The actual progress_callback is a deep closure inside run_sync; reproducing
# the full gateway turn here is impractical. These tests pin the dispatch
# CONTRACT the branch implements (which event_type → which queue card, per
# mode) so a future refactor that diverges from it fails loudly. The branch
# body itself is integration-smoke-tested separately.

class TestSubagentDispatchContract:
    def _make_dispatcher(self):
        """Build a stateful mirror of the gateway/run.py subagent branch.

        Returns a `dispatch(mode, event_type, ...)` closure that carries the
        same `last_subagent_id` header-dedup state the real callback closure
        holds, so multi-event sequences exercise the dedup contract.

        DRIFT WARNING: this is a hand-copied mirror of the production branch in
        gateway/run.py progress_callback (the `if event_type in {"subagent.tool",
        ...}` block). It is faithful as written but is NOT wired to the real
        closure — if you edit the production branch's mode/header-dedup logic,
        update this mirror in lockstep or these tests will pass against stale
        behavior. The formatter tests above DO call the real
        _format_subagent_tool_card, so card-rendering drift is caught there.
        """
        from gateway.run import (
            _format_subagent_tool_card,
            _format_subagent_progress_card,
        )

        last_subagent_id = [None]

        def dispatch(mode, event_type, tool_name=None, preview=None, **kwargs):
            out = []
            if event_type in {
                "subagent.tool",
                "subagent.progress",
                "subagent_progress",
                "subagent.tool_completed",
            }:
                if mode == "full" and event_type == "subagent.tool":
                    _sub_id = kwargs.get("subagent_id")
                    _include_header = (not _sub_id) or (_sub_id != last_subagent_id[0])
                    card = _format_subagent_tool_card(
                        tool_name,
                        preview,
                        goal=kwargs.get("goal"),
                        task_index=int(kwargs.get("task_index", 0) or 0),
                        task_count=int(kwargs.get("task_count", 1) or 1),
                        include_header=_include_header,
                    )
                    if card:
                        if _sub_id:
                            last_subagent_id[0] = _sub_id
                        out.append(
                            (
                                "__tool_start__",
                                "__subagent__",
                                card,
                                _sub_id or "",
                                str(tool_name or ""),
                            )
                        )
                elif mode == "full" and event_type == "subagent.tool_completed":
                    out.append(
                        (
                            "__subagent_duration__",
                            kwargs.get("subagent_id") or "",
                            str(tool_name or ""),
                            kwargs.get("duration"),
                            bool(kwargs.get("is_error", False)),
                        )
                    )
                elif mode == "batched" and event_type in {
                    "subagent.progress",
                    "subagent_progress",
                }:
                    card = _format_subagent_progress_card(preview or tool_name)
                    if card:
                        out.append(("__tool_start__", "__subagent__", card))
                return ("CONSUMED", out)
            return ("PASSTHROUGH", out)

        return dispatch

    def _dispatch(self, mode, event_type, tool_name=None, preview=None, **kwargs):
        """Single-shot dispatch (fresh state each call)."""
        return self._make_dispatcher()(
            mode, event_type, tool_name, preview, **kwargs
        )

    def test_off_drops_all_subagent_events(self):
        consumed, out = self._dispatch(
            "off", "subagent.tool", "terminal", "ls", goal="g"
        )
        assert consumed == "CONSUMED" and out == []
        consumed, out = self._dispatch("off", "subagent.progress", preview="🔀 a, b")
        assert out == []

    def test_full_renders_per_tool_only(self):
        _, out = self._dispatch(
            "full", "subagent.tool", "terminal", "pytest", goal="Run tests"
        )
        assert len(out) == 1
        assert out[0][:2] == ("__tool_start__", "__subagent__")
        assert "terminal" in out[0][2] and "Run tests" in out[0][2]
        # full ignores the batched summary event
        _, out2 = self._dispatch("full", "subagent.progress", preview="🔀 a, b, c")
        assert out2 == []

    def test_full_dedups_header_for_same_subagent(self):
        # Consecutive tool calls from the SAME subagent_id: header on the first,
        # bare tool line on the rest. Mirrors the consecutive-terminal-block
        # header drop.
        dispatch = self._make_dispatcher()
        sid = "sa-0-abc123"
        _, out1 = dispatch(
            "full", "subagent.tool", "terminal", "pytest",
            goal="Run the suite", subagent_id=sid,
        )
        _, out2 = dispatch(
            "full", "subagent.tool", "read_file", "config.yaml",
            goal="Run the suite", subagent_id=sid,
        )
        _, out3 = dispatch(
            "full", "subagent.tool", "search_files", "TODO",
            goal="Run the suite", subagent_id=sid,
        )
        # First carries the header; subsequent ones do not.
        assert "🔀" in out1[0][2] and "Run the suite" in out1[0][2]
        assert "🔀" not in out2[0][2] and "Run the suite" not in out2[0][2]
        assert "🔀" not in out3[0][2]
        # But each still shows its own tool + preview.
        assert "read_file" in out2[0][2] and "config.yaml" in out2[0][2]
        assert "search_files" in out3[0][2]

    def test_full_reemits_header_when_subagent_changes(self):
        # Parallel batch: switching subagent_id re-emits the header so the user
        # can tell which child is acting.
        dispatch = self._make_dispatcher()
        _, out_a1 = dispatch(
            "full", "subagent.tool", "terminal", "a",
            goal="Child A", subagent_id="sa-0-aaa", task_index=0, task_count=2,
        )
        _, out_b1 = dispatch(
            "full", "subagent.tool", "read_file", "b",
            goal="Child B", subagent_id="sa-1-bbb", task_index=1, task_count=2,
        )
        _, out_a2 = dispatch(
            "full", "subagent.tool", "search_files", "a2",
            goal="Child A", subagent_id="sa-0-aaa", task_index=0, task_count=2,
        )
        assert "🔀" in out_a1[0][2] and "[1]" in out_a1[0][2]
        # Different subagent → header re-emitted, with its own index tag.
        assert "🔀" in out_b1[0][2] and "[2]" in out_b1[0][2]
        # Switched back to A → header re-emitted (last_subagent_id was B).
        assert "🔀" in out_a2[0][2]

    def test_full_header_always_when_subagent_id_missing(self):
        # Fail-safe: no subagent_id means we can't dedup, so every card keeps
        # its header rather than silently stripping the only context line.
        dispatch = self._make_dispatcher()
        _, out1 = dispatch("full", "subagent.tool", "terminal", "x", goal="G")
        _, out2 = dispatch("full", "subagent.tool", "read_file", "y", goal="G")
        assert "🔀" in out1[0][2]
        assert "🔀" in out2[0][2]

    def test_full_interleaved_subagents_reemit_every_header(self):
        # Two concurrent children whose events interleave (A,B,A,B). Dedup is a
        # single scalar, so every card re-emits its header — degrades to the
        # prior always-header behavior. Never wrong-drops, never crashes.
        dispatch = self._make_dispatcher()
        cards = []
        for i in range(4):
            sid = "sa-0-aaa" if i % 2 == 0 else "sa-1-bbb"
            goal = "Child A" if i % 2 == 0 else "Child B"
            _, out = dispatch(
                "full", "subagent.tool", "read_file", f"f{i}",
                goal=goal, subagent_id=sid, task_index=(i % 2), task_count=2,
            )
            cards.append(out[0][2])
        # Every interleaved card kept its header (no silent header loss).
        assert all("🔀" in c for c in cards)

    def test_cards_use_subagent_sentinel_not_delegate_task(self):
        # Regression guard for nit N2: subagent cards must NOT be queued under
        # the "delegate_task" tool name, or they'd absorb the parent's
        # delegate_task completion-duration suffix via the pending-tool FIFO.
        _, out = self._dispatch("full", "subagent.tool", "terminal", "ls", goal="g")
        assert out[0][1] == "__subagent__"
        assert out[0][1] != "delegate_task"

    def test_batched_renders_summary_only(self):
        _, out = self._dispatch(
            "batched", "subagent.tool", "terminal", "pytest", goal="g"
        )
        assert out == []
        _, out2 = self._dispatch(
            "batched", "subagent.progress", preview="🔀 terminal, read_file"
        )
        assert len(out2) == 1 and "terminal, read_file" in out2[0][2]

    def test_batched_renders_depth2_underscore_event(self):
        # Depth-2 grandchild summary: legacy underscore event with the summary
        # in the tool_name slot, preview empty.
        _, out = self._dispatch(
            "batched", "subagent_progress", tool_name="🔀 [1] web_search, read_file"
        )
        assert len(out) == 1
        assert "web_search, read_file" in out[0][2]

    def test_non_subagent_event_passes_through(self):
        verdict, out = self._dispatch("full", "tool.started", "read_file")
        assert verdict == "PASSTHROUGH" and out == []
        verdict, _ = self._dispatch("full", "tool.completed", "read_file")
        assert verdict == "PASSTHROUGH"

    def test_full_tool_completed_enqueues_duration(self):
        # subagent.tool_completed → "__subagent_duration__" tuple carrying
        # (sub_id, tool, duration, is_error). batched/off ignore it.
        _, out = self._dispatch(
            "full", "subagent.tool_completed", "terminal",
            subagent_id="sa-0-aaa", duration=0.012, is_error=False,
        )
        assert len(out) == 1
        assert out[0][0] == "__subagent_duration__"
        assert out[0][1] == "sa-0-aaa"
        assert out[0][2] == "terminal"
        assert out[0][3] == 0.012
        assert out[0][4] is False

    def test_batched_and_off_ignore_tool_completed(self):
        _, out_b = self._dispatch(
            "batched", "subagent.tool_completed", "terminal",
            subagent_id="sa-0-aaa", duration=0.012,
        )
        assert out_b == []
        _, out_o = self._dispatch(
            "off", "subagent.tool_completed", "terminal",
            subagent_id="sa-0-aaa", duration=0.012,
        )
        assert out_o == []

    def test_start_tuple_carries_subagent_id_and_tool(self):
        # The 5-tuple start card must carry sub_id + real tool so the consumer
        # can build its (sub_id, tool) pending key.
        _, out = self._dispatch(
            "full", "subagent.tool", "read_file", "x.py",
            goal="g", subagent_id="sa-1-bbb",
        )
        assert out[0][0] == "__tool_start__"
        assert out[0][1] == "__subagent__"
        assert out[0][3] == "sa-1-bbb"
        assert out[0][4] == "read_file"


# ---------------------------------------------------------------------------
# run._append_subagent_duration_to_card — benchmark suffix placement
# ---------------------------------------------------------------------------

class TestAppendSubagentDurationToCard:
    def test_suffix_lands_on_tool_line_not_header(self):
        from gateway.run import _append_subagent_duration_to_card

        card = '🔀 Review the diff\n└ 💻 terminal  "pytest"'
        out = _append_subagent_duration_to_card(card, 0.012, is_error=False)
        lines = out.splitlines()
        # Header untouched; suffix on the tool (last) line.
        assert lines[0] == "🔀 Review the diff"
        assert lines[1].endswith("· 12ms")
        assert "·" not in lines[0]

    def test_suffix_on_bare_deduped_card(self):
        from gateway.run import _append_subagent_duration_to_card

        card = '└ ⚙️ read_file  "config.yaml"'
        out = _append_subagent_duration_to_card(card, 1.5)
        assert out.endswith("· 1.5s")
        assert out.count("\n") == 0

    def test_error_suffix(self):
        from gateway.run import _append_subagent_duration_to_card

        out = _append_subagent_duration_to_card("└ terminal", 2.0, is_error=True)
        assert "failed after 2.0s" in out


# ---------------------------------------------------------------------------
# End-to-end consumer mirror: out-of-order parallel duration matching
# ---------------------------------------------------------------------------
# This models the gateway queue consumer's progress_lines + (sub_id, tool)
# pending map and proves the headline correctness property: with two parallel
# children running the SAME tool name, completions arriving OUT OF ORDER each
# land on their own child's card — no cross-attribution. Same drift caveat as
# the dispatch harness: hand-copied mirror of the consumer branch.

class TestSubagentDurationMatching:
    def _make_consumer(self):
        from gateway.run import (
            _format_subagent_tool_card,
            _append_subagent_duration_to_card,
        )

        progress_lines = []
        subagent_pending = {}  # (sub_id, tool) -> [indexes]
        last_subagent_id = [None]

        def on_start(tool, preview, goal, sub_id, idx=0, cnt=1):
            inc = (not sub_id) or (sub_id != last_subagent_id[0])
            card = _format_subagent_tool_card(
                tool, preview, goal=goal, task_index=idx, task_count=cnt,
                include_header=inc,
            )
            if sub_id:
                last_subagent_id[0] = sub_id
            progress_lines.append(card)
            subagent_pending.setdefault((sub_id or "", tool), []).append(
                len(progress_lines) - 1
            )

        def on_complete(tool, sub_id, duration, is_error=False):
            key = (sub_id or "", tool)
            idxs = subagent_pending.get(key) or []
            while idxs:
                i = idxs.pop(0)
                if 0 <= i < len(progress_lines):
                    progress_lines[i] = _append_subagent_duration_to_card(
                        progress_lines[i], duration, is_error=is_error
                    )
                    return i
            return None

        return progress_lines, on_start, on_complete

    def test_out_of_order_parallel_no_cross_attribution(self):
        lines, on_start, on_complete = self._make_consumer()
        # Two children both run read_file; A starts first, B second.
        on_start("read_file", "a.py", "Child A", "sa-0-aaa", 0, 2)  # line 0
        on_start("read_file", "b.py", "Child B", "sa-1-bbb", 1, 2)  # line 1
        # Completions arrive OUT OF ORDER: B finishes first, then A.
        idx_b = on_complete("read_file", "sa-1-bbb", 0.005)
        idx_a = on_complete("read_file", "sa-0-aaa", 2.0)
        assert idx_b == 1 and idx_a == 0
        # Each duration landed on its OWN child's card.
        assert "a.py" in lines[0] and "· 2.0s" in lines[0]
        assert "b.py" in lines[1] and "· 5ms" in lines[1]
        # No cross-contamination.
        assert "· 5ms" not in lines[0]
        assert "· 2.0s" not in lines[1]

    def test_same_child_repeated_tool_fifo(self):
        lines, on_start, on_complete = self._make_consumer()
        # One child runs read_file twice; FIFO matching within the child.
        on_start("read_file", "first.py", "Child A", "sa-0-aaa")   # line 0
        on_start("read_file", "second.py", "Child A", "sa-0-aaa")  # line 1 (deduped header)
        on_complete("read_file", "sa-0-aaa", 0.1)   # → oldest = line 0
        on_complete("read_file", "sa-0-aaa", 0.2)   # → next = line 1
        assert "first.py" in lines[0] and "· 0.1s" in lines[0]
        assert "second.py" in lines[1] and "· 0.2s" in lines[1]

    def test_unmatched_completion_is_dropped_no_orphan(self):
        lines, on_start, on_complete = self._make_consumer()
        # Completion with no matching start card → returns None, no row added.
        result = on_complete("terminal", "sa-9-zzz", 0.5)
        assert result is None
        assert lines == []  # no orphan "✅ completed" line
