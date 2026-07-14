import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cli as cli_module
from cli import HermesCLI


class _FakeBuffer:
    def __init__(self, text="", cursor_position=None):
        self.text = text
        self.cursor_position = len(text) if cursor_position is None else cursor_position

    def reset(self, append_to_history=False):
        self.text = ""
        self.cursor_position = 0


def _make_cli_stub():
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_state = None
    cli._approval_deadline = 0
    cli._approval_lock = threading.Lock()
    cli._sudo_state = None
    cli._sudo_deadline = 0
    cli._modal_input_snapshot = None
    cli._invalidate = MagicMock()
    cli._app = SimpleNamespace(invalidate=MagicMock(), current_buffer=_FakeBuffer())
    return cli


def _make_background_cli_stub():
    cli = _make_cli_stub()
    cli._background_task_counter = 0
    cli._background_tasks = {}
    cli._ensure_runtime_credentials = MagicMock(return_value=True)
    cli._resolve_turn_agent_config = MagicMock(return_value={
        "model": "test-model",
        "runtime": {
            "api_key": "test-key",
            "base_url": "https://example.test/v1",
            "provider": "test",
            "api_mode": "chat_completions",
        },
        "request_overrides": None,
    })
    cli.max_turns = 90
    cli.enabled_toolsets = []
    cli._session_db = None
    cli.reasoning_config = {}
    cli.service_tier = None
    cli._providers_only = None
    cli._providers_ignore = None
    cli._providers_order = None
    cli._provider_sort = None
    cli._provider_require_params = None
    cli._provider_data_collection = None
    cli._openrouter_min_coding_score = None
    cli._fallback_model = None
    cli._agent_running = False
    cli._spinner_text = ""
    cli.bell_on_complete = False
    cli.final_response_markdown = "strip"
    return cli


class TestCliApprovalUi:
    def test_smart_denied_callback_offers_only_once_and_deny(self):
        cli = _make_cli_stub()
        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback(
                "rm -rf /tmp/example",
                "recursive delete",
                allow_permanent=False,
                smart_denied=True,
            )

        thread = threading.Thread(target=_run_callback, daemon=True)
        thread.start()

        deadline = time.time() + 2
        while cli._approval_state is None and time.time() < deadline:
            time.sleep(0.01)

        assert cli._approval_state is not None
        assert cli._approval_state["choices"] == ["once", "deny"]

        cli._approval_state["response_queue"].put("deny")
        thread.join(timeout=2)
        assert result["value"] == "deny"

    def test_non_smart_non_permanent_callback_preserves_session_choice(self):
        cli = _make_cli_stub()
        assert cli._approval_choices(
            "rm -rf /tmp/example", allow_permanent=False, smart_denied=False
        ) == ["once", "session", "deny"]

    def test_sudo_prompt_restores_existing_draft_after_response(self):
        cli = _make_cli_stub()
        cli._app.current_buffer = _FakeBuffer("draft command", cursor_position=5)
        result = {}

        def _run_callback():
            result["value"] = cli._sudo_password_callback()

        with patch.object(cli_module, "_cprint"):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()

            deadline = time.time() + 2
            while cli._sudo_state is None and time.time() < deadline:
                time.sleep(0.01)

            assert cli._sudo_state is not None
            assert cli._app.current_buffer.text == ""

            cli._app.current_buffer.text = "secret"
            cli._app.current_buffer.cursor_position = len("secret")
            cli._sudo_state["response_queue"].put("secret")

            thread.join(timeout=2)

        assert result["value"] == "secret"
        assert cli._app.current_buffer.text == "draft command"
        assert cli._app.current_buffer.cursor_position == 5

    def test_approval_callback_includes_view_for_long_commands(self):
        cli = _make_cli_stub()
        command = "sudo dd if=/tmp/githubcli-keyring.gpg of=/usr/share/keyrings/githubcli-archive-keyring.gpg bs=4M status=progress"
        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback(command, "disk copy")

        thread = threading.Thread(target=_run_callback, daemon=True)
        thread.start()

        deadline = time.time() + 2
        while cli._approval_state is None and time.time() < deadline:
            time.sleep(0.01)

        assert cli._approval_state is not None
        assert "view" in cli._approval_state["choices"]

        cli._approval_state["response_queue"].put("deny")
        thread.join(timeout=2)
        assert result["value"] == "deny"

    def test_handle_approval_selection_view_expands_in_place(self):
        cli = _make_cli_stub()
        cli._approval_state = {
            "command": "sudo dd if=/tmp/in of=/usr/share/keyrings/githubcli-archive-keyring.gpg bs=4M status=progress",
            "description": "disk copy",
            "choices": ["once", "session", "always", "deny", "view"],
            "selected": 4,
            "response_queue": queue.Queue(),
        }

        cli._handle_approval_selection()

        assert cli._approval_state is not None
        assert cli._approval_state["show_full"] is True
        assert "view" not in cli._approval_state["choices"]
        assert cli._approval_state["selected"] == 3
        assert cli._approval_state["response_queue"].empty()

    def test_approval_display_places_title_inside_box_not_border(self):
        cli = _make_cli_stub()
        cli._approval_state = {
            "command": "sudo dd if=/tmp/in of=/usr/share/keyrings/githubcli-archive-keyring.gpg bs=4M status=progress",
            "description": "disk copy",
            "choices": ["once", "session", "always", "deny", "view"],
            "selected": 0,
            "response_queue": queue.Queue(),
        }

        fragments = cli._get_approval_display_fragments()
        rendered = "".join(text for _style, text in fragments)
        lines = rendered.splitlines()

        assert lines[0].startswith("╭")
        assert "Dangerous Command" not in lines[0]
        assert any("Dangerous Command" in line for line in lines[1:3])
        assert "Show full command" in rendered
        assert "githubcli-archive-" in rendered
        assert "keyring.gpg" in rendered
        assert "status=progress" in rendered

    def test_approval_display_wraps_preview_hint_on_narrow_terminal(self):
        cli = _make_cli_stub()
        cli._approval_state = {
            "command": "sudo " + ("very-long-command-segment-" * 8),
            "description": "shell command via -c/-lc flag",
            "choices": ["once", "session", "always", "deny", "view"],
            "selected": 0,
            "response_queue": queue.Queue(),
        }

        import shutil as _shutil

        with patch("cli.shutil.get_terminal_size",
                   return_value=_shutil.os.terminal_size((30, 24))):
            fragments = cli._get_approval_display_fragments()

        rendered = "".join(text for _style, text in fragments)
        lines = rendered.splitlines()
        border_width = len(lines[0])

        assert "Show full" in rendered
        assert "command)" in rendered
        assert all(len(line) == border_width for line in lines)

    def test_approval_display_shows_full_command_after_view(self):
        cli = _make_cli_stub()
        full_command = "sudo dd if=/tmp/in of=/usr/share/keyrings/githubcli-archive-keyring.gpg bs=4M status=progress"
        cli._approval_state = {
            "command": full_command,
            "description": "disk copy",
            "choices": ["once", "session", "always", "deny"],
            "selected": 0,
            "show_full": True,
            "response_queue": queue.Queue(),
        }

        fragments = cli._get_approval_display_fragments()
        rendered = "".join(text for _style, text in fragments)

        assert "..." not in rendered
        assert "githubcli-" in rendered
        assert "archive-" in rendered
        assert "keyring.gpg" in rendered
        assert "status=progress" in rendered

    def test_approval_display_preserves_command_and_choices_with_long_description(self):
        """Regression: long tirith descriptions used to push approve/deny off-screen.

        The panel must always render the command and every choice, even when
        the description would otherwise wrap into 10+ lines. The description
        gets truncated with a marker instead.
        """
        cli = _make_cli_stub()
        long_desc = (
            "Security scan — [CRITICAL] Destructive shell command with wildcard expansion: "
            "The command performs a recursive deletion of log files which may contain "
            "audit information relevant to active incident investigations, running services "
            "that rely on log files for state, rotated archives, and other system artifacts. "
            "Review whether this is intended before approving. Consider whether a targeted "
            "deletion with more specific filters would better match the intent."
        )
        cli._approval_state = {
            "command": "rm -rf /var/log/apache2/*.log",
            "description": long_desc,
            "choices": ["once", "session", "always", "deny"],
            "selected": 0,
            "response_queue": queue.Queue(),
        }

        # Simulate a compact terminal where the old unbounded panel would overflow.
        import shutil as _shutil

        with patch("cli.shutil.get_terminal_size",
                   return_value=_shutil.os.terminal_size((100, 20))):
            fragments = cli._get_approval_display_fragments()

        rendered = "".join(text for _style, text in fragments)

        # Command must be fully visible (rm -rf /var/log/apache2/*.log is short).
        assert "rm -rf /var/log/apache2/*.log" in rendered

        # Every choice must render — this is the core bug: approve/deny were
        # getting clipped off the bottom of the panel.
        assert "Allow once" in rendered
        assert "Allow for this session" in rendered
        assert "Add to permanent allowlist" in rendered
        assert "Deny" in rendered

        # The bottom border must render (i.e. the panel is self-contained).
        assert rendered.rstrip().endswith("╯")

        # The description gets truncated — marker should appear.
        assert "(description truncated)" in rendered

    def test_approval_display_skips_description_on_very_short_terminal(self):
        """On a 12-row terminal, only the command and choices have room.

        The description is dropped entirely rather than partially shown, so the
        choices never get clipped.
        """
        cli = _make_cli_stub()
        cli._approval_state = {
            "command": "rm -rf /var/log/apache2/*.log",
            "description": "recursive delete",
            "choices": ["once", "session", "always", "deny"],
            "selected": 0,
            "response_queue": queue.Queue(),
        }

        import shutil as _shutil

        with patch("cli.shutil.get_terminal_size",
                   return_value=_shutil.os.terminal_size((100, 12))):
            fragments = cli._get_approval_display_fragments()

        rendered = "".join(text for _style, text in fragments)

        # Command visible.
        assert "rm -rf /var/log/apache2/*.log" in rendered
        # All four choices visible.
        for label in ("Allow once", "Allow for this session",
                      "Add to permanent allowlist", "Deny"):
            assert label in rendered, f"choice {label!r} missing"

    def test_approval_display_truncates_giant_command_in_view_mode(self):
        """If the user hits /view on a massive command, choices still render.

        The command gets truncated with a marker; the description gets dropped
        if there's no remaining row budget.
        """
        cli = _make_cli_stub()
        # 50 lines of command when wrapped at ~64 chars.
        giant_cmd = "bash -c 'echo " + ("x" * 3000) + "'"
        cli._approval_state = {
            "command": giant_cmd,
            "description": "shell command via -c/-lc flag",
            "choices": ["once", "session", "always", "deny"],
            "selected": 0,
            "show_full": True,
            "response_queue": queue.Queue(),
        }

        import shutil as _shutil

        with patch("cli.shutil.get_terminal_size",
                   return_value=_shutil.os.terminal_size((100, 24))):
            fragments = cli._get_approval_display_fragments()

        rendered = "".join(text for _style, text in fragments)

        # All four choices visible even with a huge command.
        for label in ("Allow once", "Allow for this session",
                      "Add to permanent allowlist", "Deny"):
            assert label in rendered, f"choice {label!r} missing"

        # Command got truncated with a marker.
        assert "(command truncated" in rendered

    def test_background_task_registers_thread_local_approval_callbacks(self):
        """Background /btw tasks must use the prompt_toolkit approval UI.

        The foreground chat path registers dangerous-command callbacks inside
        its worker thread because tools.terminal_tool stores them in
        threading.local(). /background used to skip that, so dangerous commands
        fell back to raw input() in a background thread and timed out under
        prompt_toolkit.
        """
        cli = _make_background_cli_stub()
        seen = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                self._print_fn = None
                self.thinking_callback = None

            def run_conversation(self, **kwargs):
                from tools.terminal_tool import (
                    _get_approval_callback,
                    _get_sudo_password_callback,
                )

                seen["approval"] = _get_approval_callback()
                seen["sudo"] = _get_sudo_password_callback()
                return {
                    "final_response": "done",
                    "messages": [],
                    "completed": True,
                    "failed": False,
                }

        with patch.object(cli_module, "AIAgent", FakeAgent), \
             patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "ChatConsole") as chat_console:
            chat_console.return_value.print = MagicMock()
            cli._handle_background_command("/btw check weather")

            # Join the worker thread deterministically rather than polling a
            # wall-clock deadline — under load the thread's finally-block pop
            # of _background_tasks can lag a fixed timeout, which flaked CI.
            for _thread in list(cli._background_tasks.values()):
                _thread.join(timeout=10)

        assert seen["approval"].__self__ is cli
        assert seen["approval"].__func__ is HermesCLI._approval_callback
        assert seen["sudo"].__self__ is cli
        assert seen["sudo"].__func__ is HermesCLI._sudo_password_callback
        assert not cli._background_tasks


def _make_real_paint_cli_stub():
    """A stub whose modal repaint path runs the REAL _paint_now / _invalidate.

    Both gates are set adversarially: _resize_recovery_pending=True and a recent
    _last_invalidate inside the throttle window. A throttled _invalidate() would
    be dropped under these conditions — _paint_now must paint regardless.
    """
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_state = None
    cli._approval_deadline = 0
    cli._approval_lock = threading.Lock()
    cli._sudo_state = None
    cli._sudo_deadline = 0
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._clarify_deadline = 0
    cli._modal_input_snapshot = None
    # Real methods, not mocks.
    cli._paint_now = HermesCLI._paint_now.__get__(cli, HermesCLI)
    cli._invalidate = HermesCLI._invalidate.__get__(cli, HermesCLI)
    cli._resize_recovery_pending = True       # gate 1: resize in flight
    cli._last_invalidate = time.monotonic()   # gate 2: inside throttle window
    cli._app = SimpleNamespace(invalidate=MagicMock(), current_buffer=_FakeBuffer())
    return cli


class TestModalPaintNow:
    """Regression for #41098 — modal prompts must paint immediately.

    The dangerous-command approval, clarify, and sudo prompts run their wait
    loop on a background thread, set modal state a ConditionalContainer reads,
    then must repaint so the panel becomes visible. They used the throttled
    _invalidate(), whose paint is silently dropped on a 250ms window collision
    or while a resize is pending — so the prompt timed out unseen. They now use
    _paint_now(), which paints directly like the modal key-binding handlers.
    """

    def test_paint_now_bypasses_throttle_and_resize_guard(self):
        cli = _make_real_paint_cli_stub()
        # A bare _invalidate() is suppressed under both gates...
        cli._invalidate()
        assert not cli._app.invalidate.called
        # ...but _paint_now() always paints.
        cli._paint_now()
        assert cli._app.invalidate.called

    def test_paint_now_no_app_is_safe(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli._app = None
        cli._paint_now()  # must not raise

    def _drive(self, cli, target, state_attr):
        result = {}

        def _run():
            result["value"] = target()

        with patch.object(cli_module, "_cprint"):
            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            deadline = time.time() + 2
            while getattr(cli, state_attr) is None and time.time() < deadline:
                time.sleep(0.01)
            assert getattr(cli, state_attr) is not None
            assert cli._app.invalidate.called, (
                f"{state_attr} panel was not painted despite throttle + resize gates"
            )
            # Reset so we can prove the response-received teardown also repaints
            # (the panel must clear at once, not be held by the throttle).
            cli._app.invalidate.reset_mock()
            getattr(cli, state_attr)["response_queue"].put(
                "deny" if state_attr == "_approval_state" else
                ("a" if state_attr == "_clarify_state" else "pw")
            )
            thread.join(timeout=2)
            # clarify returns immediately on a response (no teardown repaint);
            # approval and sudo repaint to tear the panel down.
            if state_attr != "_clarify_state":
                assert cli._app.invalidate.called, (
                    f"{state_attr} panel was not repainted on teardown"
                )
        assert not thread.is_alive()
        return result["value"]

    def test_approval_prompt_paints_under_both_gates(self):
        cli = _make_real_paint_cli_stub()
        value = self._drive(
            cli, lambda: cli._approval_callback("rm -rf /tmp/scratch", "danger"),
            "_approval_state",
        )
        assert value == "deny"

    def test_clarify_prompt_paints_under_both_gates(self):
        cli = _make_real_paint_cli_stub()
        value = self._drive(
            cli, lambda: cli._clarify_callback("Pick one", ["a", "b"]),
            "_clarify_state",
        )
        assert value == "a"

    def test_sudo_prompt_paints_under_both_gates(self):
        cli = _make_real_paint_cli_stub()
        value = self._drive(cli, cli._sudo_password_callback, "_sudo_state")
        assert value == "pw"

    def test_secret_response_teardown_paints(self):
        """_submit_secret_response tears the secret panel down via _paint_now,
        so the panel clears immediately rather than being held by the throttle."""
        cli = _make_real_paint_cli_stub()
        cli._secret_state = {"response_queue": queue.Queue()}
        cli._secret_deadline = 0
        cli._submit_secret_response("hunter2")
        assert cli._secret_state is None
        assert cli._app.invalidate.called
        assert cli._secret_state is None  # cleared


class TestApprovalCallbackThreadLocalWiring:
    """Regression guard for the thread-local callback freeze (#13617 / #13618).

    After 62348cff made _approval_callback / _sudo_password_callback thread-local
    (ACP GHSA-qg5c-hvr5-hjgr), the CLI agent thread could no longer see callbacks
    registered in the main thread — the dangerous-command prompt silently fell
    back to stdin input() and deadlocked against prompt_toolkit. The fix is to
    register the callbacks INSIDE the agent worker thread (matching the ACP
    pattern). These tests lock in that invariant.
    """

    def test_main_thread_registration_is_invisible_to_child_thread(self):
        """Confirms the underlying threading.local semantics that drove the bug.

        If this ever starts passing as "visible", the thread-local isolation
        is gone and the ACP race GHSA-qg5c-hvr5-hjgr may be back.
        """
        from tools.terminal_tool import (
            set_approval_callback,
            _get_approval_callback,
        )

        def main_cb(_cmd, _desc):
            return "once"

        set_approval_callback(main_cb)
        try:
            seen = {}

            def _child():
                seen["value"] = _get_approval_callback()

            t = threading.Thread(target=_child, daemon=True)
            t.start()
            t.join(timeout=2)
            assert seen["value"] is None
        finally:
            set_approval_callback(None)

    def test_child_thread_registration_is_visible_and_cleared_in_finally(self):
        """The fix pattern: register INSIDE the worker thread, clear in finally.

        This is exactly what cli.py's run_agent() closure does. If this test
        fails, the CLI approval prompt freeze (#13617) has regressed.
        """
        from tools.terminal_tool import (
            set_approval_callback,
            set_sudo_password_callback,
            _get_approval_callback,
            _get_sudo_password_callback,
        )

        def approval_cb(_cmd, _desc):
            return "once"

        def sudo_cb():
            return "hunter2"

        seen = {}

        def _worker():
            # Mimic cli.py's run_agent() thread target.
            set_approval_callback(approval_cb)
            set_sudo_password_callback(sudo_cb)
            try:
                seen["approval"] = _get_approval_callback()
                seen["sudo"] = _get_sudo_password_callback()
            finally:
                set_approval_callback(None)
                set_sudo_password_callback(None)
                seen["approval_after"] = _get_approval_callback()
                seen["sudo_after"] = _get_sudo_password_callback()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=2)

        assert seen["approval"] is approval_cb
        assert seen["sudo"] is sudo_cb
        # Finally block must clear both slots — otherwise a reused thread
        # would hold a stale reference to a disposed CLI instance.
        assert seen["approval_after"] is None
        assert seen["sudo_after"] is None


class TestPersistPromptSummary:
    """display.persist_prompts — one-line scrollback record of resolved modals."""

    def _resolve_approval(self, cli, answer, command="rm -rf /tmp/scratch"):
        result = {}

        def _run():
            result["value"] = cli._approval_callback(command, "danger")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        deadline = time.time() + 2
        while cli._approval_state is None and time.time() < deadline:
            time.sleep(0.01)
        cli._approval_state["response_queue"].put(answer)
        t.join(timeout=2)
        return result["value"]

    def test_approval_resolution_prints_summary_line(self):
        cli = _make_cli_stub()
        printed = []
        with patch.object(cli_module, "_cprint", printed.append):
            verdict = self._resolve_approval(cli, "session")
        assert verdict == "session"
        summary = "\n".join(printed)
        assert "Approval" in summary
        assert "rm -rf /tmp/scratch" in summary
        assert "allowed for session" in summary

    def test_approval_summary_truncates_long_command(self):
        cli = _make_cli_stub()
        printed = []
        long_cmd = "sudo " + ("x" * 300)
        with patch.object(cli_module, "_cprint", printed.append):
            self._resolve_approval(cli, "deny", command=long_cmd)
        summary = "\n".join(printed)
        assert "denied" in summary
        assert "…" in summary
        # The raw 300-char tail must not be dumped wholesale.
        assert "x" * 200 not in summary

    def test_persist_prompts_false_suppresses_summary(self):
        cli = _make_cli_stub()
        printed = []
        with patch.dict(cli_module.CLI_CONFIG.get("display", {}), {"persist_prompts": False}), \
             patch.object(cli_module, "_cprint", printed.append):
            verdict = self._resolve_approval(cli, "once")
        assert verdict == "once"
        assert not any("Approval" in p for p in printed)

    def test_clarify_resolution_prints_summary_line(self):
        cli = _make_cli_stub()
        cli._clarify_state = None
        cli._clarify_freetext = False
        cli._clarify_deadline = 0
        printed = []
        result = {}

        def _run():
            result["value"] = cli._clarify_callback("Pick a path?", ["A", "B"])

        with patch.object(cli_module, "_cprint", printed.append):
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            deadline = time.time() + 2
            while cli._clarify_state is None and time.time() < deadline:
                time.sleep(0.01)
            cli._clarify_state["response_queue"].put("B")
            t.join(timeout=2)

        assert result["value"] == "B"
        summary = "\n".join(printed)
        assert "Clarify" in summary
        assert "Pick a path?" in summary
        assert "B" in summary


class TestClearOverlaysForInterrupt:
    """Regression tests for #14026 — interrupting a running agent must clear
    every input-blocking overlay (approval/clarify/sudo/secret) so the CLI
    isn't left frozen with no thread servicing the prompt."""

    def _make_cli(self):
        cli = _make_cli_stub()
        # Attributes the helper touches that the base stub doesn't set.
        cli._clarify_state = None
        cli._clarify_freetext = False
        cli._secret_state = None
        cli._secret_deadline = 0
        cli._paint_now = MagicMock()
        return cli

    def test_clears_all_four_overlays_and_unblocks_queues(self):
        cli = self._make_cli()
        approval_q = queue.Queue()
        clarify_q = queue.Queue()
        sudo_q = queue.Queue()
        secret_q = queue.Queue()
        cli._approval_state = {"response_queue": approval_q}
        cli._clarify_state = {"response_queue": clarify_q}
        cli._clarify_freetext = True
        cli._sudo_state = {"response_queue": sudo_q, "timeout": 60}
        cli._sudo_deadline = 99999.0
        cli._secret_state = {"response_queue": secret_q, "var_name": "X"}

        cli._clear_active_overlays_for_interrupt()

        # All states nilled out.
        assert cli._approval_state is None
        assert cli._clarify_state is None
        assert cli._clarify_freetext is False
        assert cli._sudo_state is None
        assert cli._sudo_deadline == 0
        assert cli._secret_state is None

        # Each blocked thread would have received a terminal value.
        assert approval_q.get_nowait() == "deny"
        assert clarify_q.get_nowait()  # cancellation sentinel string
        assert sudo_q.get_nowait() == ""
        assert secret_q.get_nowait() == ""

    def test_noop_when_no_overlays_active(self):
        cli = self._make_cli()
        cli._clear_active_overlays_for_interrupt()
        assert cli._approval_state is None
        assert cli._clarify_state is None
        assert cli._sudo_state is None
        assert cli._secret_state is None

    def test_dead_queue_does_not_block_clearing_others(self):
        """A queue that raises on put() must not prevent the remaining
        overlays from being cleared."""
        cli = self._make_cli()

        class _DeadQueue:
            def put(self, *_a, **_k):
                raise RuntimeError("queue gone")

        clarify_q = queue.Queue()
        cli._approval_state = {"response_queue": _DeadQueue()}
        cli._clarify_state = {"response_queue": clarify_q}

        cli._clear_active_overlays_for_interrupt()

        assert cli._approval_state is None  # cleared despite dead queue
        assert cli._clarify_state is None
        assert clarify_q.get_nowait()

    def test_interrupt_unblocks_thread_blocked_on_approval(self):
        """End-to-end: a worker blocked on the approval queue unblocks when the
        interrupt helper drains it."""
        cli = self._make_cli()
        approval_q = queue.Queue()
        cli._approval_state = {"response_queue": approval_q}
        result = {}

        def _worker():
            result["value"] = approval_q.get(timeout=2)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        time.sleep(0.05)
        cli._clear_active_overlays_for_interrupt()
        t.join(timeout=2)

        assert not t.is_alive(), "worker thread never unblocked"
        assert result["value"] == "deny"


class TestApprovalRemoteControl:
    """Mobile deny/extend wiring for the dangerous-command approval prompt.

    Approval itself stays local-only: the store only ever drives deny/extend,
    never a remote approve.
    """

    def _wait_for_state(self, cli, timeout=3.0):
        deadline = time.time() + timeout
        while cli._approval_state is None and time.time() < deadline:
            time.sleep(0.01)
        assert cli._approval_state is not None

    def test_approval_callback_accepts_remote_deny(self):
        cli = _make_cli_stub()
        cli._remote_intervention_settings = lambda: {
            "enabled": True,
            "allow_deny": True,
            "allow_extend": True,
            "max_total_wait_minutes": 15,
            "risk_explanation": {
                "enabled": False,
                "only_for_risk_levels": ["high", "critical"],
            },
        }
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="4242")
        )
        cli._rc_cleanup = MagicMock(return_value=0)

        calls = {"n": 0}

        def _consume(code):
            calls["n"] += 1
            assert code == "4242"
            if calls["n"] < 2:
                return None
            return SimpleNamespace(
                decision="deny",
                decision_source="telegram",
                deadline_ts=time.time() + 60,
            )

        cli._rc_consume = MagicMock(side_effect=_consume)

        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback("rm -rf /tmp/x", "wipe")

        with patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "notify_human_intervention", create=True):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            thread.join(timeout=4)

        assert not thread.is_alive()
        assert result["value"] == "deny"
        cli._rc_create_pending.assert_called_once()

    def test_approval_callback_remote_extend_bumps_deadline(self):
        cli = _make_cli_stub()
        cli._remote_intervention_settings = lambda: {
            "enabled": True,
            "allow_deny": True,
            "allow_extend": True,
            "max_total_wait_minutes": 15,
            "risk_explanation": {
                "enabled": False,
                "only_for_risk_levels": ["high", "critical"],
            },
        }
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="7777")
        )
        cli._rc_cleanup = MagicMock(return_value=0)

        calls = {"n": 0}

        def _consume(code):
            calls["n"] += 1
            if calls["n"] == 1:
                return SimpleNamespace(
                    decision="extend",
                    decision_source="telegram",
                    deadline_ts=time.time() + 600,
                )
            return None

        cli._rc_consume = MagicMock(side_effect=_consume)

        result = {}
        response_queue_holder = {}

        def _run_callback():
            result["value"] = cli._approval_callback("rm -rf /tmp/x", "wipe")

        with patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "notify_human_intervention", create=True):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()

            self._wait_for_state(cli)
            initial_deadline = cli._approval_deadline
            response_queue_holder["q"] = cli._approval_state["response_queue"]

            # Wait until the extend is consumed and the deadline bumps.
            deadline = time.time() + 4
            while (cli._approval_deadline <= initial_deadline + 1
                   and time.time() < deadline):
                time.sleep(0.02)

            assert cli._approval_deadline > initial_deadline + 1, \
                "extend should have increased the monotonic deadline"
            # Still waiting (not returned) after the extend.
            assert thread.is_alive()
            assert "value" not in result

            response_queue_holder["q"].put("once")
            thread.join(timeout=3)

        assert not thread.is_alive()
        assert result["value"] == "once"

    def test_local_answer_wins_over_remote(self):
        cli = _make_cli_stub()
        cli._remote_intervention_settings = lambda: {
            "enabled": True,
            "allow_deny": True,
            "allow_extend": True,
            "max_total_wait_minutes": 15,
            "risk_explanation": {
                "enabled": False,
                "only_for_risk_levels": ["high", "critical"],
            },
        }
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="5555")
        )
        cli._rc_cleanup = MagicMock(return_value=0)
        cli._rc_consume = MagicMock(
            return_value=SimpleNamespace(
                decision="deny",
                decision_source="telegram",
                deadline_ts=time.time() + 60,
            )
        )

        # Put the local answer on the queue BEFORE creating the prompt so the
        # very first response_queue.get() succeeds — local must win.
        pre_queue = queue.Queue()
        pre_queue.put("session")

        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback("rm -rf /tmp/x", "wipe")

        # Inject our pre-filled queue by patching queue.Queue used in the
        # callback to hand back our prepared queue exactly once.
        orig_queue_cls = queue.Queue
        made = {"n": 0}

        def _queue_factory(*a, **kw):
            made["n"] += 1
            if made["n"] == 1:
                return pre_queue
            return orig_queue_cls(*a, **kw)

        with patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "notify_human_intervention", create=True), \
             patch.object(cli_module.queue, "Queue", side_effect=_queue_factory):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            thread.join(timeout=3)

        assert not thread.is_alive()
        assert result["value"] == "session"

    def test_remote_approve_is_ignored(self):
        cli = _make_cli_stub()
        cli._remote_intervention_settings = lambda: {
            "enabled": True,
            "allow_deny": True,
            "allow_extend": True,
            "max_total_wait_minutes": 15,
            "risk_explanation": {
                "enabled": False,
                "only_for_risk_levels": ["high", "critical"],
            },
        }
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="9090")
        )
        cli._rc_cleanup = MagicMock(return_value=0)
        cli._rc_consume = MagicMock(return_value=None)

        captured = {}

        def _capture_notify(*args, **kwargs):
            captured.update(kwargs)

        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback("rm -rf /tmp/x", "wipe")

        # The callback imports notify_human_intervention from its source
        # module, so patch it there (not on cli_module).
        import hermes_cli.human_intervention_notifications as notif_mod
        with patch.object(cli_module, "_cprint"), \
             patch.object(notif_mod, "notify_human_intervention",
                          side_effect=_capture_notify):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            self._wait_for_state(cli)
            cli._approval_state["response_queue"].put("deny")
            thread.join(timeout=3)

        assert not thread.is_alive()
        assert result["value"] == "deny"
        actions = captured.get("remote_actions")
        assert actions is not None
        assert "approve" not in actions
        assert "deny" in actions

    def test_high_risk_calls_explainer_low_risk_does_not(self):
        cli = _make_cli_stub()
        rc = {
            "risk_explanation": {
                "enabled": True,
                "only_for_risk_levels": ["high", "critical"],
                "max_chars": 280,
                "timeout_seconds": 3,
            }
        }
        high = cli._explain_command_risk_for_notify(
            "rm -rf /x", "d", "high", rc
        )
        assert isinstance(high, str)
        assert high != ""

        low = cli._explain_command_risk_for_notify(
            "ls -la", "d", "low", rc
        )
        assert low == ""

    def test_remote_disabled_creates_no_pending(self):
        cli = _make_cli_stub()
        cli._remote_intervention_settings = lambda: {
            "enabled": False,
            "allow_deny": True,
            "allow_extend": True,
            "max_total_wait_minutes": 15,
            "risk_explanation": {
                "enabled": False,
                "only_for_risk_levels": ["high", "critical"],
            },
        }
        cli._rc_create_pending = MagicMock()
        cli._rc_cleanup = MagicMock(return_value=0)
        cli._rc_consume = MagicMock(return_value=None)

        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback("rm -rf /tmp/x", "wipe")

        with patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "notify_human_intervention", create=True):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            self._wait_for_state(cli)
            cli._approval_state["response_queue"].put("once")
            thread.join(timeout=3)

        assert not thread.is_alive()
        assert result["value"] == "once"
        cli._rc_create_pending.assert_not_called()

    def test_approval_poll_errors_do_not_abort_local_prompt(self):
        cli = _make_cli_stub()
        cli._remote_intervention_settings = lambda: {
            "enabled": True,
            "allow_deny": True,
            "allow_extend": True,
            "max_total_wait_minutes": 15,
            "risk_explanation": {
                "enabled": False,
                "only_for_risk_levels": ["high", "critical"],
            },
        }
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="3131")
        )
        cli._rc_cleanup = MagicMock(return_value=0)
        calls = {"n": 0}

        def _flaky_consume(_code):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("store temporarily unavailable")
            return None

        cli._rc_consume = MagicMock(side_effect=_flaky_consume)

        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback("rm -rf /tmp/x", "wipe")

        with patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "notify_human_intervention", create=True):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            self._wait_for_state(cli)
            deadline = time.time() + 4
            while calls["n"] < 1 and time.time() < deadline:
                time.sleep(0.02)
            cli._approval_state["response_queue"].put("once")
            thread.join(timeout=3)

        assert calls["n"] >= 1
        assert not thread.is_alive()
        assert result["value"] == "once"

    def test_poll_remote_intervention_handles_store_errors(self):
        cli = _make_cli_stub()
        cli._rc_consume = MagicMock(side_effect=OSError("store unavailable"))
        current_deadline = time.monotonic() + 30

        signal, source, deadline = cli._poll_remote_intervention("3131", current_deadline)

        assert signal is None
        assert source == ""
        assert deadline == current_deadline

    def test_poll_remote_intervention_denies_and_extends_safely(self):
        cli = _make_cli_stub()
        current_deadline = time.monotonic() + 30
        later_wall_deadline = time.time() + 120
        cli._rc_consume = MagicMock(return_value=SimpleNamespace(
            decision="extend",
            decision_source="telegram",
            deadline_ts=later_wall_deadline,
        ))

        signal, source, deadline = cli._poll_remote_intervention("3131", current_deadline)

        assert signal == "extend"
        assert source == "telegram"
        assert deadline > current_deadline

        cli._rc_consume = MagicMock(return_value=SimpleNamespace(
            decision="deny",
            decision_source="telegram",
        ))
        signal, source, deadline2 = cli._poll_remote_intervention("3131", deadline)
        assert signal == "deny"
        assert source == "telegram"
        assert deadline2 == deadline

    def test_poll_remote_intervention_rejects_bad_or_non_extend_decisions(self):
        cli = _make_cli_stub()
        current_deadline = time.monotonic() + 30

        cli._rc_consume = MagicMock(return_value=SimpleNamespace(
            decision="extend",
            deadline_ts="not-a-number",
        ))
        assert cli._poll_remote_intervention("3131", current_deadline) == (
            None, "", current_deadline
        )

        cli._rc_consume = MagicMock(return_value=SimpleNamespace(
            decision="approve",
            deadline_ts=time.time() + 120,
        ))
        assert cli._poll_remote_intervention("3131", current_deadline) == (
            None, "", current_deadline
        )



def _make_sudo_clarify_stub():
    """CLI stub wired for sudo + clarify remote-control tests.

    Adds the clarify state slots the callbacks expect and swaps the modal
    snapshot helpers for mocks so no real prompt_toolkit app is touched.
    """
    cli = _make_cli_stub()
    cli._clarify_state = None
    cli._clarify_deadline = 0
    cli._clarify_freetext = False
    cli._capture_modal_input_snapshot = MagicMock()
    cli._restore_modal_input_snapshot = MagicMock()
    return cli


_DENY_EXTEND_RC = {
    "enabled": True,
    "allow_deny": True,
    "allow_extend": True,
    "max_total_wait_minutes": 15,
    "risk_explanation": {
        "enabled": False,
        "only_for_risk_levels": ["high", "critical"],
    },
}


class TestSudoClarifyRemoteControl:
    """Mobile deny/extend wiring for the sudo-password and clarify prompts.

    HARD SAFETY: sudo must NEVER accept a password remotely (deny == cancel,
    returns ""), and clarify must NEVER have an option chosen remotely (deny ==
    timeout-equivalent best-judgement string). Remote can only deny or extend.
    """

    def _wait_for_sudo_state(self, cli, timeout=3.0):
        deadline = time.time() + timeout
        while cli._sudo_state is None and time.time() < deadline:
            time.sleep(0.01)
        assert cli._sudo_state is not None

    def _wait_for_clarify_state(self, cli, timeout=3.0):
        deadline = time.time() + timeout
        while cli._clarify_state is None and time.time() < deadline:
            time.sleep(0.01)
        assert cli._clarify_state is not None

    def test_sudo_remote_cancel_returns_empty_never_password(self):
        cli = _make_sudo_clarify_stub()
        cli._remote_intervention_settings = lambda: dict(_DENY_EXTEND_RC)
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="5555")
        )
        cli._rc_cleanup = MagicMock(return_value=0)

        calls = {"n": 0}

        def _consume(code):
            calls["n"] += 1
            assert code == "5555"
            if calls["n"] < 2:
                return None
            return SimpleNamespace(
                decision="deny",
                decision_source="telegram",
                deadline_ts=time.time() + 45,
            )

        cli._rc_consume = MagicMock(side_effect=_consume)

        result = {}

        def _run_callback():
            result["value"] = cli._sudo_password_callback()

        import hermes_cli.human_intervention_notifications as notif_mod
        with patch.object(cli_module, "_cprint"), \
             patch.object(notif_mod, "notify_human_intervention",
                          create=True):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            # Remote deny lands on ~2nd poll (~2s) — far before the 45s timeout.
            thread.join(timeout=4)

        assert not thread.is_alive()
        # The cardinal guarantee: a remote cancel NEVER yields a password.
        assert result["value"] == ""
        cli._rc_create_pending.assert_called_once()
        cli._restore_modal_input_snapshot.assert_called()

    def test_sudo_remote_extend_bumps_deadline(self):
        cli = _make_sudo_clarify_stub()
        cli._remote_intervention_settings = lambda: dict(_DENY_EXTEND_RC)
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="6666")
        )
        cli._rc_cleanup = MagicMock(return_value=0)

        calls = {"n": 0}

        def _consume(code):
            calls["n"] += 1
            if calls["n"] == 1:
                return SimpleNamespace(
                    decision="extend",
                    decision_source="telegram",
                    deadline_ts=time.time() + 600,
                )
            return None

        cli._rc_consume = MagicMock(side_effect=_consume)

        result = {}

        def _run_callback():
            result["value"] = cli._sudo_password_callback()

        import hermes_cli.human_intervention_notifications as notif_mod
        with patch.object(cli_module, "_cprint"), \
             patch.object(notif_mod, "notify_human_intervention",
                          create=True):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()

            self._wait_for_sudo_state(cli)
            initial_deadline = cli._sudo_deadline

            deadline = time.time() + 4
            while (cli._sudo_deadline <= initial_deadline + 1
                   and time.time() < deadline):
                time.sleep(0.02)

            assert cli._sudo_deadline > initial_deadline + 1, \
                "extend should have increased the monotonic deadline"
            assert thread.is_alive()
            assert "value" not in result

            # Finish locally with a real password — local always wins.
            cli._sudo_state["response_queue"].put("hunter2")
            thread.join(timeout=3)

        assert not thread.is_alive()
        assert result["value"] == "hunter2"

    def test_clarify_remote_deny_returns_best_judgement(self):
        cli = _make_sudo_clarify_stub()
        cli._remote_intervention_settings = lambda: dict(_DENY_EXTEND_RC)
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="7070")
        )
        cli._rc_cleanup = MagicMock(return_value=0)

        calls = {"n": 0}

        def _consume(code):
            calls["n"] += 1
            assert code == "7070"
            if calls["n"] < 2:
                return None
            return SimpleNamespace(
                decision="deny",
                decision_source="telegram",
                deadline_ts=time.time() + 120,
            )

        cli._rc_consume = MagicMock(side_effect=_consume)

        result = {}

        def _run_callback():
            result["value"] = cli._clarify_callback("pick one", ["a", "b"])

        import hermes_cli.human_intervention_notifications as notif_mod
        with patch.object(cli_module, "_cprint"), \
             patch.object(notif_mod, "notify_human_intervention",
                          create=True):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            thread.join(timeout=4)

        assert not thread.is_alive()
        # Remote deny == timeout-equivalent: agent decides, NO option selected.
        assert "best judgement" in result["value"]
        assert result["value"] not in ("a", "b")
        cli._rc_create_pending.assert_called_once()

    def test_clarify_remote_extend_bumps_deadline(self):
        cli = _make_sudo_clarify_stub()
        cli._remote_intervention_settings = lambda: dict(_DENY_EXTEND_RC)
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="8080")
        )
        cli._rc_cleanup = MagicMock(return_value=0)

        calls = {"n": 0}

        def _consume(code):
            calls["n"] += 1
            if calls["n"] == 1:
                return SimpleNamespace(
                    decision="extend",
                    decision_source="telegram",
                    deadline_ts=time.time() + 600,
                )
            return None

        cli._rc_consume = MagicMock(side_effect=_consume)

        result = {}

        def _run_callback():
            result["value"] = cli._clarify_callback("pick one", ["a", "b"])

        import hermes_cli.human_intervention_notifications as notif_mod
        with patch.object(cli_module, "_cprint"), \
             patch.object(notif_mod, "notify_human_intervention",
                          create=True):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()

            self._wait_for_clarify_state(cli)
            initial_deadline = cli._clarify_deadline

            deadline = time.time() + 4
            while (cli._clarify_deadline <= initial_deadline + 1
                   and time.time() < deadline):
                time.sleep(0.02)

            assert cli._clarify_deadline > initial_deadline + 1, \
                "extend should have increased the monotonic deadline"
            assert thread.is_alive()
            assert "value" not in result

            cli._clarify_state["response_queue"].put("a")
            thread.join(timeout=3)

        assert not thread.is_alive()
        assert result["value"] == "a"

    def test_sudo_remote_disabled_no_pending(self):
        cli = _make_sudo_clarify_stub()
        cli._remote_intervention_settings = lambda: {
            "enabled": False,
            "allow_deny": True,
            "allow_extend": True,
            "max_total_wait_minutes": 15,
            "risk_explanation": {
                "enabled": False,
                "only_for_risk_levels": ["high", "critical"],
            },
        }
        cli._rc_create_pending = MagicMock()
        cli._rc_cleanup = MagicMock(return_value=0)
        cli._rc_consume = MagicMock(return_value=None)

        result = {}

        def _run_callback():
            result["value"] = cli._sudo_password_callback()

        import hermes_cli.human_intervention_notifications as notif_mod
        with patch.object(cli_module, "_cprint"), \
             patch.object(notif_mod, "notify_human_intervention",
                          create=True):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            self._wait_for_sudo_state(cli)
            cli._sudo_state["response_queue"].put("pw")
            thread.join(timeout=3)

        assert not thread.is_alive()
        assert result["value"] == "pw"
        cli._rc_create_pending.assert_not_called()


_APPROVE_RC = {
    "enabled": True,
    "allow_approve": True,
    "approve_medium": True,
    "approve_high_typed_confirm": True,
    "approve_token_len": 4,
    "allow_deny": True,
    "allow_extend": True,
    "max_total_wait_minutes": 15,
    "never_approve_levels": ["critical"],
    "risk_explanation": {
        "enabled": False,
        "use_llm": False,
        "only_for_risk_levels": ["high", "critical"],
    },
}


class TestApprovalRemoteApprove:
    """Phase-2 mobile APPROVE wiring for the dangerous-command prompt.

    medium → one-tap, high → typed-confirm, critical → never approvable.
    Local answer still wins; deny/extend behaviour is unchanged.
    """

    def _wait_for_state(self, cli, timeout=3.0):
        deadline = time.time() + timeout
        while cli._approval_state is None and time.time() < deadline:
            time.sleep(0.01)
        assert cli._approval_state is not None

    def test_approval_medium_remote_one_tap_returns_once(self):
        cli = _make_cli_stub()
        cli._remote_intervention_settings = lambda: dict(_APPROVE_RC)
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="4242", approve_token="")
        )
        cli._rc_cleanup = MagicMock(return_value=0)

        calls = {"n": 0}

        def _consume(code):
            calls["n"] += 1
            assert code == "4242"
            if calls["n"] < 2:
                return None
            return SimpleNamespace(
                decision="approve",
                decision_source="telegram",
                deadline_ts=time.time() + 60,
                approve_token="",
            )

        cli._rc_consume = MagicMock(side_effect=_consume)

        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback(
                "ls", "desc", risk_level="medium"
            )

        with patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "notify_human_intervention", create=True):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            thread.join(timeout=4)

        assert not thread.is_alive()
        assert result["value"] == "once"
        cli._rc_create_pending.assert_called_once()
        kwargs = cli._rc_create_pending.call_args.kwargs
        assert kwargs.get("approve_tier") == "one_tap"
        assert kwargs.get("risk_level") == "medium"

    def test_approval_high_remote_typed_confirm_returns_once(self):
        cli = _make_cli_stub()
        cli._remote_intervention_settings = lambda: dict(_APPROVE_RC)
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="4242", approve_token="4815")
        )
        cli._rc_cleanup = MagicMock(return_value=0)

        calls = {"n": 0}

        def _consume(code):
            calls["n"] += 1
            assert code == "4242"
            if calls["n"] < 2:
                return None
            return SimpleNamespace(
                decision="approve",
                decision_source="telegram",
                deadline_ts=time.time() + 60,
                approve_token="4815",
            )

        cli._rc_consume = MagicMock(side_effect=_consume)

        captured = {}

        def _capture_notify(*args, **kwargs):
            captured.update(kwargs)

        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback(
                "rm -rf /tmp/x", "wipe", risk_level="high"
            )

        import hermes_cli.human_intervention_notifications as notif_mod
        with patch.object(cli_module, "_cprint"), \
             patch.object(notif_mod, "notify_human_intervention",
                          side_effect=_capture_notify):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            thread.join(timeout=4)

        assert not thread.is_alive()
        assert result["value"] == "once"
        cli._rc_create_pending.assert_called_once()
        kwargs = cli._rc_create_pending.call_args.kwargs
        assert kwargs.get("approve_tier") == "typed_confirm"
        # The generated token flows into the notification.
        assert captured.get("approve_tier") == "typed_confirm"
        assert captured.get("approve_token") == "4815"

    def test_compute_approve_tier_levels(self):
        cli = _make_cli_stub()
        rc = dict(_APPROVE_RC)

        tier, never = cli._compute_approve_tier("medium", rc)
        assert tier == "one_tap"
        assert "critical" in never

        tier, _ = cli._compute_approve_tier("high", rc)
        assert tier == "typed_confirm"

        tier, _ = cli._compute_approve_tier("critical", rc)
        assert tier == "none"

        tier, _ = cli._compute_approve_tier("low", rc)
        assert tier == "none"

        # allow_approve disabled → never approvable, regardless of level.
        rc_off = dict(_APPROVE_RC)
        rc_off["allow_approve"] = False
        assert cli._compute_approve_tier("medium", rc_off)[0] == "none"
        assert cli._compute_approve_tier("high", rc_off)[0] == "none"

    def test_approval_critical_no_remote_approve(self):
        cli = _make_cli_stub()
        cli._remote_intervention_settings = lambda: dict(_APPROVE_RC)
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="4242", approve_token="")
        )
        cli._rc_cleanup = MagicMock(return_value=0)
        cli._rc_consume = MagicMock(return_value=None)

        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback(
                "rm -rf /", "wipe root", risk_level="critical"
            )

        with patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "notify_human_intervention", create=True):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            self._wait_for_state(cli)
            cli._approval_state["response_queue"].put("deny")
            thread.join(timeout=3)

        assert not thread.is_alive()
        assert result["value"] == "deny"
        cli._rc_create_pending.assert_called_once()
        kwargs = cli._rc_create_pending.call_args.kwargs
        assert kwargs.get("approve_tier") == "none"

    def test_approval_remote_deny_still_returns_deny(self):
        cli = _make_cli_stub()
        cli._remote_intervention_settings = lambda: dict(_APPROVE_RC)
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="4242", approve_token="")
        )
        cli._rc_cleanup = MagicMock(return_value=0)

        calls = {"n": 0}

        def _consume(code):
            calls["n"] += 1
            if calls["n"] < 2:
                return None
            return SimpleNamespace(
                decision="deny",
                decision_source="telegram",
                deadline_ts=time.time() + 60,
                approve_token="",
            )

        cli._rc_consume = MagicMock(side_effect=_consume)

        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback(
                "rm -rf /tmp/x", "wipe", risk_level="medium"
            )

        with patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "notify_human_intervention", create=True):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            thread.join(timeout=4)

        assert not thread.is_alive()
        assert result["value"] == "deny"

    def test_local_answer_wins_over_remote_approve(self):
        cli = _make_cli_stub()
        cli._remote_intervention_settings = lambda: dict(_APPROVE_RC)
        cli._rc_create_pending = MagicMock(
            return_value=SimpleNamespace(code="5555", approve_token="")
        )
        cli._rc_cleanup = MagicMock(return_value=0)
        cli._rc_consume = MagicMock(
            return_value=SimpleNamespace(
                decision="approve",
                decision_source="telegram",
                deadline_ts=time.time() + 60,
                approve_token="",
            )
        )

        # Local answer queued BEFORE the prompt → first get() wins.
        pre_queue = queue.Queue()
        pre_queue.put("session")

        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback(
                "rm -rf /tmp/x", "wipe", risk_level="medium"
            )

        orig_queue_cls = queue.Queue
        made = {"n": 0}

        def _queue_factory(*a, **kw):
            made["n"] += 1
            if made["n"] == 1:
                return pre_queue
            return orig_queue_cls(*a, **kw)

        with patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "notify_human_intervention", create=True), \
             patch.object(cli_module.queue, "Queue", side_effect=_queue_factory):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()
            thread.join(timeout=3)

        assert not thread.is_alive()
        assert result["value"] == "session"

    def test_explain_uses_llm_when_configured(self):
        cli = _make_cli_stub()
        import hermes_cli.human_intervention_risk_explainer as expl_mod

        sentinel = object()
        captured = {}

        def _capture_explain(**kwargs):
            captured.update(kwargs)
            return "explained"

        rc_on = {
            "risk_explanation": {
                "enabled": True,
                "use_llm": True,
                "only_for_risk_levels": ["high", "critical"],
                "max_chars": 280,
                "timeout_seconds": 3,
            }
        }
        with patch.object(expl_mod, "default_llm_fn", sentinel, create=True), \
             patch.object(expl_mod, "explain_command_risk",
                          side_effect=_capture_explain):
            out = cli._explain_command_risk_for_notify(
                "rm -rf /x", "d", "high", rc_on
            )
        assert out == "explained"
        assert captured.get("llm_fn") is sentinel

        # use_llm False → llm_fn stays None.
        captured.clear()
        rc_off = {
            "risk_explanation": {
                "enabled": True,
                "use_llm": False,
                "only_for_risk_levels": ["high", "critical"],
                "max_chars": 280,
                "timeout_seconds": 3,
            }
        }
        with patch.object(expl_mod, "default_llm_fn", sentinel, create=True), \
             patch.object(expl_mod, "explain_command_risk",
                          side_effect=_capture_explain):
            cli._explain_command_risk_for_notify("rm -rf /x", "d", "high", rc_off)
        assert captured.get("llm_fn") is None
