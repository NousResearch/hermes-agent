"""Regression tests for loading feedback on slow slash commands."""

import threading
import time
from unittest.mock import patch

import pytest

from cli import HermesCLI


class TestCLILoadingIndicator:
    def _make_cli(self):
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj._app = None
        cli_obj._last_invalidate = 0.0
        cli_obj._command_running = False
        cli_obj._command_status = ""
        return cli_obj

    @staticmethod
    def _join_reload_worker(cli_obj):
        """Never leak a background reload worker out of a test."""
        worker = getattr(cli_obj, "_mcp_reload_thread", None)
        if worker is not None:
            worker.join(timeout=10)
            assert not worker.is_alive(), "background reload worker did not exit"

    def test_skills_command_sets_busy_state_and_prints_status(self, capsys):
        cli_obj = self._make_cli()
        seen = {}

        def fake_handle(cmd: str):
            seen["cmd"] = cmd
            seen["running"] = cli_obj._command_running
            seen["status"] = cli_obj._command_status
            print("skills done")

        with patch.object(cli_obj, "_handle_skills_command", side_effect=fake_handle), \
             patch.object(cli_obj, "_invalidate") as invalidate_mock:
            assert cli_obj.process_command("/skills search kubernetes")

        output = capsys.readouterr().out
        assert "⏳ Searching skills..." in output
        assert "skills done" in output
        assert seen == {
            "cmd": "/skills search kubernetes",
            "running": True,
            "status": "Searching skills...",
        }
        assert cli_obj._command_running is False
        assert cli_obj._command_status == ""
        assert invalidate_mock.call_count == 2

    def test_reload_mcp_does_not_reserve_the_composer(self, capsys):
        """/reload-mcp must not take the blocking busy state.

        It used to run ``_reload_mcp()`` inside ``_busy_command(...)`` on the
        command thread.  That context manager sets ``_command_blocks_input``,
        so a slow MCP server froze the session for as long as the reload took.
        The reload now runs on a daemon worker and reports itself, so the
        command claims no busy state at all.
        """
        cli_obj = self._make_cli()
        seen = {}
        done = threading.Event()

        def fake_reload():
            seen["running"] = cli_obj._command_running
            seen["blocks_input"] = getattr(cli_obj, "_command_blocks_input", False)
            print("reload done")
            done.set()

        # Pre-approve via config so the handler goes straight to the reload
        # rather than through the confirmation modal.
        fake_cfg = {"approvals": {"mcp_reload_confirm": False}}

        with patch.object(cli_obj, "_reload_mcp", side_effect=fake_reload), \
             patch.object(cli_obj, "_invalidate") as invalidate_mock, \
             patch("cli.load_cli_config", return_value=fake_cfg):
            assert cli_obj.process_command("/reload-mcp")
            assert done.wait(timeout=5), "background reload worker never ran"
            self._join_reload_worker(cli_obj)

        output = capsys.readouterr().out
        assert "⏳ Reloading MCP servers..." not in output
        assert "reload done" in output
        # The worker never sees (or sets) the input-blocking busy state.
        assert seen == {"running": False, "blocks_input": False}
        assert cli_obj._command_running is False
        assert cli_obj._command_status == ""
        assert invalidate_mock.call_count == 0

    @pytest.mark.parametrize("route", ["pre_approved", "modal_once"])
    def test_reload_mcp_returns_before_reload_completes(self, route):
        """Both approved routes hand off and return while the reload is stuck.

        Covers the pre-approved route (``approvals.mcp_reload_confirm: false``)
        and the modal "Approve Once" route.  Fixing only one of them would
        leave the other frozen.
        """
        cli_obj = self._make_cli()
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def blocking_reload():
            started.set()
            release.wait(timeout=10)
            finished.set()

        confirm_required = route == "modal_once"
        fake_cfg = {"approvals": {"mcp_reload_confirm": confirm_required}}

        try:
            with patch.object(cli_obj, "_reload_mcp", side_effect=blocking_reload), \
                 patch.object(cli_obj, "_invalidate"), \
                 patch.object(cli_obj, "_prompt_text_input_modal", return_value="once"), \
                 patch("cli.load_cli_config", return_value=fake_cfg):
                begin = time.monotonic()
                assert cli_obj.process_command("/reload-mcp")
                elapsed = time.monotonic() - begin

                # The reload really is running — we did not simply skip it.
                assert started.wait(timeout=5), "background reload worker never ran"
                # ...and the command returned before it completed.  This is the
                # assertion a join()-based fix cannot satisfy.
                assert not finished.is_set()
                assert elapsed < 1.0, f"/reload-mcp blocked for {elapsed:.2f}s"
                assert getattr(cli_obj, "_command_blocks_input", False) is False
        finally:
            release.set()
            self._join_reload_worker(cli_obj)

    def test_reload_mcp_does_not_stack_concurrent_workers(self, capsys):
        """A second /reload-mcp while one is in flight does not start a worker.

        The synchronous call used to serialize reloads implicitly; running
        them in the background removes that, so the handoff refuses to stack.
        """
        cli_obj = self._make_cli()
        started = threading.Event()
        release = threading.Event()
        calls = []

        def blocking_reload():
            calls.append(1)
            started.set()
            release.wait(timeout=10)

        fake_cfg = {"approvals": {"mcp_reload_confirm": False}}

        try:
            with patch.object(cli_obj, "_reload_mcp", side_effect=blocking_reload), \
                 patch.object(cli_obj, "_invalidate"), \
                 patch("cli.load_cli_config", return_value=fake_cfg):
                assert cli_obj.process_command("/reload-mcp")
                assert started.wait(timeout=5), "background reload worker never ran"
                first_worker = cli_obj._mcp_reload_thread

                assert cli_obj.process_command("/reload-mcp")
                assert cli_obj._mcp_reload_thread is first_worker
                assert calls == [1]
        finally:
            release.set()
            self._join_reload_worker(cli_obj)

        assert "already running" in capsys.readouterr().out
