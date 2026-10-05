"""Active plan pointer contracts at the shared compaction boundary."""

import copy
import json
from types import SimpleNamespace

import pytest

from agent.context_compressor import ContextCompressor, _build_verbatim_user_section, _synthetic_user_row
from agent.conversation_compression import _fold_todo_snapshot, _is_real_user_message
from agent.conversation_compression_plan_pointer import _fold_plan_pointer
from agent.title_generator import is_titleable_user_message
from tests.agent.test_compression_rotation_state import (
    _build_agent_with_db, _conforming_fold, _msgs, refresh_state_db as refresh_state_db,
)

class TestPlanPointerFold:
    path = ".hermes/plans/2026-10-05_120000-x.md"
    header = "[Active plan from /plan: "

    @staticmethod
    def _plan_turn(path):
        import json
        from agent.plan_prompt import build_plan_prompt

        return [
            {"role": "user", "content": build_plan_prompt("task A")},
            {"role": "assistant", "tool_calls": [{
                "id": "plan-write", "type": "function", "function": {
                    "name": "write_file", "arguments": json.dumps({"path": path, "content": "plan"}),
                },
            }]},
            {"role": "tool", "tool_call_id": "plan-write", "content": "written"},
        ]

    @pytest.fixture
    def fold(self, refresh_state_db):
        db = refresh_state_db
        db.create_session("PLAN_POINTER", source="cli")
        agent = _build_agent_with_db(db, "PLAN_POINTER", platform="cli")

        def compress(history, tail=None):
            agent.context_compressor.compress.return_value = _conforming_fold(
                tail or {"role": "user", "content": "continue"},
            )
            messages = _msgs() + history + [{"role": "assistant", "content": "persisted answer"}]
            return agent._compress_context(messages, "sys", approx_tokens=120_000)[0]

        return agent, compress

    def _assert_pointer(self, compressed, path):
        text = "\n".join(str(row.get("content", "")) for row in compressed)
        assert text.count(self.header) == 1
        assert f"{self.header}{path}. Re-read it before continuing the planned work.]" in text

    def test_plan_write_survives_compaction(self, fold):
        _, compress = fold
        self._assert_pointer(compress(self._plan_turn(self.path)), self.path)

    @pytest.mark.parametrize("retain_pointer", [False, True])
    def test_second_compaction_has_one_pointer(self, fold, retain_pointer):
        _, compress = fold
        first = compress(self._plan_turn(self.path))
        second = compress(first, copy.deepcopy(first[-1]) if retain_pointer else None)
        self._assert_pointer(second, self.path)

    def test_newer_plan_replaces_pointer(self, fold):
        _, compress = fold
        first = compress(self._plan_turn(self.path))
        newer = ".hermes/plans/2026-10-05_130000-y.md"
        second = compress(first + self._plan_turn(newer), copy.deepcopy(first[-1]))
        self._assert_pointer(second, newer)
        assert self.path not in str(second)
        third = compress(self._plan_turn(self.path) + second)
        self._assert_pointer(third, newer)

    def test_no_plan_turn_has_no_pointer(self, fold):
        _, compress = fold
        assert self.header not in str(compress(self._plan_turn(self.path)[1:]))

    def test_write_outside_plans_has_no_pointer(self, fold):
        _, compress = fold
        assert self.header not in str(compress(self._plan_turn("notes/plan.md")))

    def test_pointer_and_todo_survive_two_compactions(self, fold):
        agent, compress = fold
        agent._todo_store.write([{"id": "t1", "content": "task A", "status": "pending"}])
        first = compress(self._plan_turn(self.path))
        second = compress(first, copy.deepcopy(first[-1]))
        self._assert_pointer(second, self.path)
        assert "task A" in str(second)


    def test_unflagged_pointer_only_row_mid_history_is_removed_not_blanked(self):
        from types import SimpleNamespace
        from agent.conversation_compression_plan_pointer import _fold_plan_pointer

        pointer = f"{self.header}{self.path}. Re-read it before continuing the planned work.]"
        agent = SimpleNamespace(_repair_message_sequence=lambda rows: None)
        compressed = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": pointer},  # flag dropped by an earlier todo strip
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "next"},
        ]
        _fold_plan_pointer(agent, [], compressed)
        assert all(row.get("content") for row in compressed if row["role"] == "user")
        self._assert_pointer(compressed, self.path)
        assert compressed[-1]["content"].startswith("next")

    @property
    def pointer(self):
        return f"{self.header}{self.path}. Re-read it before continuing the planned work.]"

    @staticmethod
    def _assert_alternating_users(rows):
        assert not any(a["role"] == b["role"] == "user" for a, b in zip(rows, rows[1:]))

    @pytest.mark.parametrize("tail_role,initial_todos", [("user", True), ("assistant", True), ("assistant", False)])
    def test_user_rows_alternate_across_todo_seam(self, fold, tail_role, initial_todos):
        agent, compress = fold
        todos = [{"id": "t1", "content": "task A", "status": "pending"}]
        if initial_todos:
            agent._todo_store.write(todos)
        tail = {"role": tail_role, "content": "continue" if tail_role == "user" else "persisted answer"}
        # Leave enough history for the no-growth guard to commit this fold.
        bulk = [{"role": "user", "content": "earlier request " + "x" * 6000},
                {"role": "assistant", "content": "earlier answer"}]
        first = compress(bulk + self._plan_turn(self.path), tail)
        self._assert_alternating_users(first)
        self._assert_pointer(first, self.path)
        agent._todo_store.write(todos)
        second = compress(first, copy.deepcopy(first[-1]))
        self._assert_alternating_users(second)
        self._assert_pointer(second, self.path)
        assert "task A" in str(second)

    def test_unflagged_pointer_and_done_todos_leave_no_empty_row(self, fold):
        agent, _ = fold
        agent._todo_store.write([{"id": "t1", "content": "task A", "status": "pending"}])
        snapshot = agent._todo_store.format_for_injection()
        agent._todo_store.write([{"id": "t1", "content": "task A", "status": "completed"}])
        rows = [{"role": "user", "content": "start"}, {"role": "assistant", "content": "ok"},
                {"role": "user", "content": self.pointer + "\n\n" + snapshot},
                {"role": "assistant", "content": "later"}, {"role": "user", "content": "continue"}]
        for _ in range(2):
            history = copy.deepcopy(rows)
            _fold_todo_snapshot(agent, rows)
            _fold_plan_pointer(agent, history, rows)
            assert all(row.get("content") for row in rows if row["role"] == "user")
            self._assert_alternating_users(rows)
            self._assert_pointer(rows, self.path)

    def test_real_user_tail_bytes_survive_repeated_folds(self):
        text = "  print(1)  \n"
        rows = [{"role": "user", "content": text}]
        agent = SimpleNamespace(_repair_message_sequence=lambda rows: None)
        history = self._plan_turn(self.path)
        for _ in range(2):
            _fold_plan_pointer(agent, history, rows)
            assert rows[-1]["content"] == text + "\n\n" + self.pointer
            assert _is_real_user_message(rows[-1])
            history = copy.deepcopy(rows)

    @pytest.mark.parametrize("embedded", [False, True])
    def test_structured_user_bytes_and_images_survive(self, embedded):
        text = {"type": "text", "text": "    print(1)\n"}
        image = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
        parts = [text, image, {"type": "text", "text": self.pointer}]
        if embedded:
            parts = [{**text, "text": text["text"] + "\n\n" + self.pointer}, image]
        rows = [{"role": "user", "content": copy.deepcopy(parts)}]
        agent = SimpleNamespace(_repair_message_sequence=lambda rows: None)
        for _ in range(2):
            _fold_plan_pointer(agent, copy.deepcopy(rows), rows)
            assert rows[-1]["content"][:2] == [text, image]
            self._assert_pointer(rows, self.path)

    def test_pointer_is_synthetic_for_focus_and_user_attribution(self):
        rows = [{"role": "user", "content": self.pointer}]
        assert ContextCompressor._derive_auto_focus_topic(rows) is None
        assert ContextCompressor._is_synthetic_compression_user_turn(rows[0])
        assert _synthetic_user_row(self.pointer)
        assert _build_verbatim_user_section(rows) == ""
        assert not _is_real_user_message(rows[0])
        assert not is_titleable_user_message(self.pointer)

    def test_later_ordinary_write_does_not_replace_plan(self):
        history = self._plan_turn(self.path) + [
            {"role": "assistant", "content": "plan saved"},
            {"role": "user", "content": "save reference notes"},
            *self._plan_turn(".hermes/plans/reference.md")[1:],
        ]
        rows = [{"role": "user", "content": "continue"}]
        agent = SimpleNamespace(_repair_message_sequence=lambda rows: None)
        _fold_plan_pointer(agent, history, rows)
        self._assert_pointer(rows, self.path)
        assert "reference.md" not in str(rows)

    def test_bridged_plan_write_yields_pointer(self):
        history = self._plan_turn(self.path)
        function = history[1]["tool_calls"][0]["function"]
        function["arguments"] = json.dumps({"calls": [{"name": "write_file", "arguments": json.loads(function["arguments"])}]})
        function["name"] = "tool_call"
        rows = [{"role": "user", "content": "continue"}]
        _fold_plan_pointer(SimpleNamespace(_repair_message_sequence=lambda rows: None), history, rows)
        self._assert_pointer(rows, self.path)

    def test_failed_plan_write_keeps_the_accepted_pointer(self):
        history = self._plan_turn(self.path)
        history[-1]["content"] = "Error: write refused"
        rows = [{"role": "user", "content": "continue"}]
        _fold_plan_pointer(SimpleNamespace(_repair_message_sequence=lambda rows: None), history, rows)
        self._assert_pointer(rows, self.path)
