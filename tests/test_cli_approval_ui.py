import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from cli import HermesCLI


def _make_cli_stub():
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_state = None
    cli._approval_deadline = 0
    cli._approval_lock = threading.Lock()
    cli._invalidate = MagicMock()
    cli._app = SimpleNamespace(invalidate=MagicMock())
    return cli


class TestCliApprovalUi:
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
        assert "githubcli-archive-keyring.gpg" not in rendered

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
