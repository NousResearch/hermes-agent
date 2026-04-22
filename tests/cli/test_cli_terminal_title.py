from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from cli import HermesCLI


class _SessionDB:
    def __init__(self, title=None):
        self._title = title

    def get_session_title(self, session_id):
        return self._title


def _make_cli(title=None, pending_title=None, spinner_text="", app_output=None):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._session_db = _SessionDB(title)
    cli_obj.session_id = "sess-1"
    cli_obj._pending_title = pending_title
    cli_obj._spinner_text = spinner_text
    cli_obj._tool_start_time = 0.0
    cli_obj._app = SimpleNamespace(output=app_output) if app_output is not None else None
    return cli_obj


def test_compose_terminal_title_prefers_session_topic_and_activity():
    cli_obj = _make_cli(title="Fix terminal tab titles", spinner_text="search files")

    assert cli_obj._compose_terminal_title() == "Fix terminal tab titles · search files — Hermes"


def test_compose_terminal_title_falls_back_to_pending_title_before_session_exists():
    cli_obj = _make_cli(title=None, pending_title="Investigate provider auth")

    assert cli_obj._compose_terminal_title() == "Investigate provider auth — Hermes"


def test_compose_terminal_title_uses_activity_when_topic_missing():
    cli_obj = _make_cli(title=None, spinner_text="running pytest")

    assert cli_obj._compose_terminal_title() == "running pytest — Hermes"


def test_update_terminal_title_writes_to_prompt_toolkit_output():
    output = MagicMock()
    cli_obj = _make_cli(title="Fix terminal tab titles", app_output=output)

    cli_obj._update_terminal_title()

    output.set_title.assert_called_once_with("Fix terminal tab titles — Hermes")
    output.flush.assert_called_once_with()


def test_on_auto_title_ready_refreshes_terminal_title_and_ui():
    cli_obj = _make_cli(title="Fix terminal tab titles")
    cli_obj._update_terminal_title = MagicMock()
    cli_obj._invalidate = MagicMock()

    cli_obj._on_auto_title_ready("Fix terminal tab titles")

    cli_obj._update_terminal_title.assert_called_once_with()
    cli_obj._invalidate.assert_called_once_with(min_interval=0.0)


def test_sync_session_id_from_agent_updates_title_and_clears_pending_title():
    cli_obj = _make_cli(title="Fix terminal tab titles", pending_title="Temporary")
    cli_obj.agent = SimpleNamespace(session_id="sess-2")
    cli_obj._update_terminal_title = MagicMock()

    assert cli_obj._sync_session_id_from_agent() is True
    assert cli_obj.session_id == "sess-2"
    assert cli_obj._pending_title is None
    cli_obj._update_terminal_title.assert_called_once_with()


def test_handle_branch_command_switches_session_without_clobbering_history(monkeypatch):
    class _SessionDB:
        def create_session(self, **_kwargs):
            return None

        def append_message(self, **_kwargs):
            return None

        def end_session(self, *_args, **_kwargs):
            return None

        def set_session_title(self, *_args, **_kwargs):
            return None

    cli_obj = HermesCLI.__new__(HermesCLI)
    original_history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    original_start = datetime(2026, 1, 1, 0, 0, 0)
    cli_obj._session_db = _SessionDB()
    cli_obj.session_id = "sess-1"
    cli_obj.session_start = original_start
    cli_obj.conversation_history = list(original_history)
    cli_obj._pending_title = None
    cli_obj._resumed = False
    cli_obj._spinner_text = ""
    cli_obj._tool_start_time = 0.0
    cli_obj.model = "test/model"
    cli_obj.max_turns = 10
    cli_obj.reasoning_config = None
    cli_obj._update_terminal_title = MagicMock()
    cli_obj.agent = SimpleNamespace(session_id="sess-1", session_start=original_start, reset_session_state=MagicMock())

    monkeypatch.setattr("cli._cprint", lambda *_args, **_kwargs: None)

    cli_obj._handle_branch_command("/branch Topic Branch")

    assert cli_obj.session_id != "sess-1"
    assert cli_obj.conversation_history == original_history
    assert cli_obj.session_start != original_start
    assert cli_obj.agent.session_start == cli_obj.session_start
    cli_obj.agent.reset_session_state.assert_called_once_with()
