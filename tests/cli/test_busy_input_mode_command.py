"""Tests for the /busy CLI command and busy-input-mode config handling."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch


def _import_cli():
    import hermes_cli.config as config_mod

    if not hasattr(config_mod, "save_env_value_secure"):
        config_mod.save_env_value_secure = lambda key, value: {
            "success": True,
            "stored_as": key,
            "validated": False,
        }

    import cli as cli_mod

    return cli_mod


class TestHandleBusyCommand(unittest.TestCase):
    def _make_cli(self, busy_input_mode="interrupt"):
        return SimpleNamespace(
            busy_input_mode=busy_input_mode,
            agent=None,
        )

    def test_no_args_shows_status(self):
        cli_mod = _import_cli()
        stub = self._make_cli("queue")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_busy_command(stub, "/busy")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("queue", printed)
        self.assertIn("interrupt", printed)

    def test_queue_argument_sets_queue_mode_and_saves(self):
        cli_mod = _import_cli()
        stub = self._make_cli("interrupt")
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_busy_command(stub, "/busy queue")

        self.assertEqual(stub.busy_input_mode, "queue")
        mock_save.assert_called_once_with("display.busy_input_mode", "queue")

    def test_interrupt_argument_sets_interrupt_mode_and_saves(self):
        cli_mod = _import_cli()
        stub = self._make_cli("queue")
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_busy_command(stub, "/busy interrupt")

        self.assertEqual(stub.busy_input_mode, "interrupt")
        mock_save.assert_called_once_with("display.busy_input_mode", "interrupt")

    def test_steer_argument_sets_steer_mode_and_saves(self):
        cli_mod = _import_cli()
        stub = self._make_cli("interrupt")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_busy_command(stub, "/busy steer")

        self.assertEqual(stub.busy_input_mode, "steer")
        mock_save.assert_called_once_with("display.busy_input_mode", "steer")
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("steer", printed.lower())

    def test_status_reports_steer_behavior(self):
        cli_mod = _import_cli()
        stub = self._make_cli("steer")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_busy_command(stub, "/busy status")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("steer", printed.lower())
        # The usage line should also advertise the steer option
        self.assertIn("steer", printed)

    def test_invalid_argument_prints_usage(self):
        cli_mod = _import_cli()
        stub = self._make_cli()
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_busy_command(stub, "/busy nonsense")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("Usage: /busy", printed)


class TestBusyCommandRegistry(unittest.TestCase):
    def test_busy_in_registry(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        names = [c.name for c in COMMAND_REGISTRY]
        assert "busy" in names

    def test_busy_subcommands_documented(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        busy = next(c for c in COMMAND_REGISTRY if c.name == "busy")
        assert busy.args_hint == "[queue|steer|interrupt|status]"
        assert busy.category == "Configuration"


class TestAfterworkCommand(unittest.TestCase):
    def test_plaintext_afterwork_phrases_are_recognized(self):
        cli_mod = _import_cli()

        for text in ("퇴근", "퇴근모드", "퇴근 모드", "afterwork", "awaymode"):
            self.assertTrue(cli_mod.HermesCLI._looks_like_afterwork_plaintext(text))

        self.assertFalse(cli_mod.HermesCLI._looks_like_afterwork_plaintext("그냥 퇴근할까?"))

    def test_plaintext_office_phrases_are_recognized(self):
        cli_mod = _import_cli()

        for text in ("출근", "출근모드", "출근 모드", "office", "morning"):
            self.assertTrue(cli_mod.HermesCLI._looks_like_office_plaintext(text))
            self.assertTrue(cli_mod.HermesCLI._should_handle_afterwork_command_inline(cli_mod.HermesCLI, text))

        self.assertFalse(cli_mod.HermesCLI._looks_like_office_plaintext("내일 출근하면"))

    def test_office_command_uses_project_mode_router(self):
        cli_mod = _import_cli()
        calls = []
        stub = SimpleNamespace(_handle_project_mode_command=lambda mode, target: calls.append((mode, target)))

        cli_mod.HermesCLI._handle_office_command(stub, "/office current")

        self.assertEqual(calls, [("office", "current")])

    def test_afterwork_all_uses_project_mode_router(self):
        cli_mod = _import_cli()
        calls = []
        stub = SimpleNamespace(_handle_project_mode_command=lambda mode, target: calls.append((mode, target)))

        cli_mod.HermesCLI._handle_afterwork_command(stub, "/afterwork all")

        self.assertEqual(calls, [("away", "all")])

    def test_commute_command_prints_help(self):
        cli_mod = _import_cli()
        printed = []

        with patch.object(cli_mod, "_cprint", lambda msg: printed.append(str(msg))):
            cli_mod.HermesCLI._handle_commute_command(SimpleNamespace())

        text = "\n".join(printed)
        self.assertIn("/projects", text)
        self.assertIn("/afterwork current", text)
        self.assertIn("/office current", text)

    def test_afterwork_steers_running_agent_without_interrupt_queue(self):
        cli_mod = _import_cli()

        class FakeAgent:
            def __init__(self):
                self.payloads = []

            def steer(self, text):
                self.payloads.append(text)
                return True

        agent = FakeAgent()
        pending = Queue()
        stub = SimpleNamespace(
            _agent_running=True,
            agent=agent,
            _pending_input=pending,
            cwd="/tmp/example-project",
        )
        stub._afterwork_prompt = lambda: cli_mod.HermesCLI._afterwork_prompt(stub)

        with patch.object(cli_mod, "_cprint"):
            cli_mod.HermesCLI._handle_afterwork_command(stub, "/afterwork")

        self.assertEqual(pending.qsize(), 0)
        self.assertEqual(len(agent.payloads), 1)
        self.assertIn("퇴근모드로 전환", agent.payloads[0])
        self.assertIn("중단하지 말고", agent.payloads[0])

    def test_afterwork_queues_when_steer_unavailable(self):
        cli_mod = _import_cli()
        pending = Queue()
        stub = SimpleNamespace(
            _agent_running=True,
            agent=None,
            _pending_input=pending,
            cwd="/tmp/example-project",
        )
        stub._afterwork_prompt = lambda: cli_mod.HermesCLI._afterwork_prompt(stub)

        with patch.object(cli_mod, "_cprint"):
            cli_mod.HermesCLI._handle_afterwork_command(stub, "퇴근모드")

        self.assertEqual(pending.qsize(), 1)
        queued = pending.get_nowait()
        self.assertIn("퇴근모드로 전환", queued)
        self.assertIn("중단하지 말고", queued)
