"""Behavioural tests for the terminal-first todo progress tray."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from cli import HermesCLI
from hermes_cli.todo_progress import (
    TodoPanelState,
    format_todo_panel_fragments,
    format_todo_plain_snapshot,
)
from tools.todo_tool import TodoStore


def _items():
    return [
        {"id": "research", "content": "Research other tools", "status": "completed"},
        {"id": "build", "content": "Add the CLI progress tray", "status": "in_progress"},
        {"id": "test", "content": "Test narrow terminal behaviour", "status": "pending"},
    ]


def _text(fragments: list[tuple[str, str]]) -> str:
    return "".join(text for _style, text in fragments)


class _FakeKeyBindings:
    def __init__(self):
        self.bindings = []

    def add(self, *keys, **_kwargs):
        def decorator(handler):
            self.bindings.append(
                SimpleNamespace(
                    keys=keys,
                    handler=handler,
                    filter=_kwargs.get("filter"),
                )
            )
            return handler

        return decorator


class TestTodoPanelState:
    def test_toggle_focuses_the_active_task_and_preserves_selection_by_id(self):
        state = TodoPanelState()

        assert state.toggle(_items()) is True
        assert state.expanded is True
        assert state.selected_id == "build"

        state.move(_items(), 1)
        assert state.selected_id == "test"

        reordered = [_items()[2], _items()[0], _items()[1]]
        state.move(reordered, 0)
        assert state.selected_id == "test"

    def test_status_change_requires_confirmation_then_updates_authoritative_store(self):
        store = TodoStore()
        store.write(_items())
        state = TodoPanelState(expanded=True, selected_id="build")

        assert state.request_status(store.read(), "completed") is True
        assert next(row for row in store.read() if row["id"] == "build")["status"] == "in_progress"
        assert "Mark" in state.notice

        assert state.confirm(store) is True
        assert next(row for row in store.read() if row["id"] == "build")["status"] == "completed"
        assert state.confirmation is None
        assert "completed" in state.notice

    def test_reopen_is_explicit_and_empty_lists_do_not_open(self):
        state = TodoPanelState()
        assert state.toggle([]) is False

        store = TodoStore()
        store.write([{"id": "done", "content": "Finished task", "status": "completed"}])
        state = TodoPanelState(expanded=True, selected_id="done")
        assert state.request_status(store.read(), "pending") is True
        assert state.confirm(store) is True
        assert store.read()[0]["status"] == "pending"

    def test_confirmation_fails_closed_when_the_task_list_changes(self):
        store = TodoStore()
        store.write([{"id": "build", "content": "Build tray", "status": "in_progress"}])
        state = TodoPanelState(expanded=True, selected_id="build")

        assert state.request_status(
            store.read(),
            "completed",
            expected_revision=store.revision,
        ) is True
        store.write(
            [{"id": "build", "content": "Deploy production", "status": "in_progress"}]
        )

        assert state.confirm(store) is False
        assert store.read()[0]["status"] == "in_progress"
        assert "changed before confirmation" in state.notice


class TestTodoPanelRendering:
    def test_compact_wide_view_is_one_clean_progress_line(self):
        output = _text(format_todo_panel_fragments(_items(), TodoPanelState(), width=80))

        assert output.count("\n") == 0
        assert "Tasks 1/3" in output
        assert "[>] Add the CLI progress tray" in output
        assert "Ctrl+T" in output
        assert "╭" not in output and "│" not in output

    def test_narrow_view_stacks_without_losing_the_active_task(self):
        output = _text(format_todo_panel_fragments(_items(), TodoPanelState(), width=32))

        assert output.splitlines() == [
            "Tasks 1/3",
            "[>] Add the CLI progress tray",
            "Ctrl+T details",
        ]

    def test_expanded_view_caps_rows_and_reports_overflow(self):
        items = [
            {"id": str(i), "content": f"Task {i}", "status": "in_progress" if i == 2 else "pending"}
            for i in range(12)
        ]
        state = TodoPanelState(expanded=True, selected_id="2")
        output = _text(format_todo_panel_fragments(items, state, width=80, max_rows=8))

        assert "Tasks 0/12" in output
        assert "> [>] Task 2" in output
        assert "... 7 more tasks" in output
        assert "m done" in output
        assert len(output.splitlines()) <= 8

    def test_expanded_view_keeps_the_selected_task_in_the_visible_window(self):
        items = [
            {"id": str(i), "content": f"Task {i}", "status": "pending"}
            for i in range(12)
        ]
        state = TodoPanelState(expanded=True, selected_id="10")

        output = _text(format_todo_panel_fragments(items, state, width=80, max_rows=8))

        assert "> [ ] Task 10" in output
        assert "... 7 more tasks" in output

    def test_completed_rows_can_be_hidden_without_changing_source_order(self):
        state = TodoPanelState(expanded=True, selected_id="build", show_completed=False)
        output = _text(format_todo_panel_fragments(_items(), state, width=80))

        assert "Research other tools" not in output
        assert output.index("Add the CLI progress tray") < output.index("Test narrow terminal behaviour")

    def test_plain_snapshot_flattens_multiline_ids_and_content(self):
        lines = format_todo_plain_snapshot([
            {"id": "task\n1", "content": "Line one\nLine two", "status": "pending"}
        ])

        assert lines[0] == "TASK task 1 PENDING Line one Line two"
        assert len(lines) == 2

    def test_plain_snapshot_strips_terminal_control_sequences(self):
        lines = format_todo_plain_snapshot([
            {
                "id": "task\x1b[2J",
                "content": "safe\x1b]0;changed-title\x07 text",
                "status": "pending",
            }
        ])

        output = "\n".join(lines)
        assert "\x1b" not in output
        assert "\x07" not in output
        assert "TASK task[2J PENDING safe]0;changed-title text" in output

    def test_plain_snapshot_is_pipe_safe_and_has_a_terminal_summary(self):
        lines = format_todo_plain_snapshot(_items())

        assert lines == [
            "TASK research COMPLETED Research other tools",
            "TASK build IN_PROGRESS Add the CLI progress tray",
            "TASK test PENDING Test narrow terminal behaviour",
            "SUMMARY completed=1 in_progress=1 pending=1 cancelled=0",
        ]
        assert all("\x1b" not in line and "\r" not in line for line in lines)


class TestCLIIntegrationSeams:
    @staticmethod
    def _bare_cli() -> HermesCLI:
        cli = HermesCLI.__new__(HermesCLI)
        cli._clarify_state = None
        cli._approval_state = None
        cli._slash_confirm_state = None
        cli._sudo_state = None
        cli._secret_state = None
        cli._model_picker_state = None
        cli._command_palette_state = None
        cli._auq_state = None
        cli._todo_panel_state = TodoPanelState()
        store = TodoStore()
        store.write(_items())
        cli.agent = SimpleNamespace(_todo_store=store)
        return cli

    def test_expanded_todo_inspector_blocks_normal_composer_navigation(self):
        cli = self._bare_cli()
        assert cli._is_normal_input_active() is True

        cli._todo_panel_state.expanded = True
        assert cli._is_normal_input_active() is False

    def test_registers_ctrl_t_as_the_todo_inspector_toggle(self):
        cli = self._bare_cli()
        cli._toggle_todo_panel = MagicMock(return_value=True)
        bindings = _FakeKeyBindings()

        cli._register_todo_tui_keybindings(bindings)

        binding = next(row for row in bindings.bindings if row.keys == ('c-t',))
        event = SimpleNamespace(app=MagicMock())
        binding.handler(event)
        cli._toggle_todo_panel.assert_called_once_with()
        event.app.invalidate.assert_called_once_with()

    def test_stash_and_todo_panels_are_mutually_exclusive(self):
        from hermes_cli.prompt_stash import PromptStash

        cli = self._bare_cli()
        cli._prompt_stash = PromptStash()
        cli._prompt_stash.stash("parked draft")
        cli._prompt_stash.panel_open = True
        bindings = _FakeKeyBindings()
        cli._register_todo_tui_keybindings(bindings)
        toggle = next(row for row in bindings.bindings if row.keys == ('c-t',))

        assert cli._stash_panel_open() is True
        assert toggle.filter() is False

        cli._prompt_stash.panel_open = False
        cli._todo_panel_state.expanded = True
        assert cli._todo_panel_open() is True

    def test_layout_places_todo_tray_immediately_above_status_bar(self):
        cli = self._bare_cli()
        cli._todo_panel_widget = "todo-panel"
        cli._stash_panel_widget = None
        cli._pet_widget = None
        cli._get_extra_tui_widgets = lambda: []

        children = cli._build_tui_layout_children(
            sudo_widget="sudo",
            secret_widget="secret",
            free_text_widget="free-text",
            approval_widget="approval",
            slash_confirm_widget="slash-confirm",
            clarify_widget="clarify",
            auq_widget="auq",
            model_picker_widget="model-picker",
            command_palette_widget="palette",
            spinner_widget="spinner",
            spacer="spacer",
            status_bar="status",
            input_rule_top="top-rule",
            image_bar="image-bar",
            input_area="input-area",
            input_rule_bot="bottom-rule",
            peer_presence_bar="peers",
            voice_status_bar="voice",
            completions_menu="completions",
        )

        assert children.index("todo-panel") + 1 == children.index("status")

    def test_toggle_keeps_the_composer_draft_in_place(self):
        cli = self._bare_cli()
        cli._capture_modal_input_snapshot = MagicMock()
        cli._restore_modal_input_snapshot = MagicMock()
        cli._invalidate = MagicMock()

        assert cli._toggle_todo_panel() is True
        assert cli._todo_panel_state.expanded is True
        assert cli._toggle_todo_panel() is True

        cli._capture_modal_input_snapshot.assert_not_called()
        cli._restore_modal_input_snapshot.assert_not_called()

    def test_empty_task_list_closes_the_inspector_before_the_next_plan(self):
        cli = self._bare_cli()
        cli._todo_panel_state.expanded = True
        assert cli.agent is not None
        cli.agent._todo_store.write([])

        assert cli._todo_panel_visible() is False
        assert cli._todo_panel_state.expanded is False
