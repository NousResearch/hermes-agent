from unittest.mock import MagicMock, patch
from typing import Any

from cli import HermesCLI


class _Buffer:
    def __init__(self):
        self.text = ""
        self.cursor_position = 0


class _App:
    def __init__(self):
        self.current_buffer = _Buffer()
        self.invalidate = MagicMock()


def _make_cli() -> Any:
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "current_session"
    cli_obj.conversation_history = []
    cli_obj.agent = None
    cli_obj.console = MagicMock()
    cli_obj._session_db = MagicMock()
    cli_obj._pending_resume_sessions = None
    cli_obj._pending_prompt_messages = None
    cli_obj._app = None
    return cli_obj


class TestCliPromptsCommand:
    def test_show_recent_prompts_includes_indexes_and_load_hint(self, capsys):
        cli_obj = _make_cli()
        cli_obj._list_recent_prompts = MagicMock(return_value=[
            {"id": 22, "timestamp": None, "preview": "second prompt"},
            {"id": 11, "timestamp": None, "preview": "first prompt"},
        ])

        prompts = cli_obj._show_recent_prompts()
        output = capsys.readouterr().out

        assert len(prompts) == 2
        assert "Recent prompts" in output
        assert "second prompt" in output
        assert "first prompt" in output
        assert "/prompts 2" in output
        assert "bare number" in output

    def test_bare_prompts_arms_pending_selection(self):
        cli_obj = _make_cli()
        prompts = [{"id": 22, "timestamp": None, "preview": "second prompt"}]
        cli_obj._show_recent_prompts = MagicMock(return_value=prompts)

        cli_obj._handle_prompts_command("/prompts")

        assert cli_obj._pending_prompt_messages == prompts

    def test_bare_prompts_no_messages_does_not_arm(self):
        cli_obj = _make_cli()
        cli_obj._show_recent_prompts = MagicMock(return_value=[])

        with patch("cli._cprint") as mock_cprint:
            cli_obj._handle_prompts_command("/prompts")

        assert cli_obj._pending_prompt_messages is None
        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert "No previous prompts" in printed

    def test_handle_prompts_by_index_prefills_selected_prompt(self):
        cli_obj = _make_cli()
        app = _App()
        cli_obj._app = app
        cli_obj._session_db.list_recent_user_messages.return_value = [
            {"id": 22, "timestamp": None, "preview": "second prompt"},
            {"id": 11, "timestamp": None, "preview": "first prompt"},
        ]
        cli_obj._session_db.get_user_message.return_value = {
            "id": 11, "role": "user", "content": "first prompt full text"
        }

        with patch("cli._cprint") as mock_cprint:
            cli_obj._handle_prompts_command("/prompts 2")

        assert app.current_buffer.text == "first prompt full text"
        assert app.current_buffer.cursor_position == len("first prompt full text")
        app.invalidate.assert_called_once()
        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert "Loaded prompt #2" in printed
        assert cli_obj._pending_prompt_messages is None

    def test_pending_number_loads_selected_prompt_and_disarms(self):
        cli_obj = _make_cli()
        app = _App()
        cli_obj._app = app
        cli_obj._pending_prompt_messages = [
            {"id": 22, "timestamp": None, "preview": "second prompt"},
            {"id": 11, "timestamp": None, "preview": "first prompt"},
        ]
        cli_obj._session_db.get_user_message.return_value = {
            "id": 11, "role": "user", "content": "first prompt full text"
        }

        with patch("cli._cprint"):
            consumed = cli_obj._consume_pending_prompt_selection("2")

        assert consumed is True
        assert app.current_buffer.text == "first prompt full text"
        assert cli_obj._pending_prompt_messages is None

    def test_pending_out_of_range_consumed_with_message(self):
        cli_obj = _make_cli()
        cli_obj._pending_prompt_messages = [
            {"id": 22, "timestamp": None, "preview": "second prompt"},
        ]

        with patch("cli._cprint") as mock_cprint:
            consumed = cli_obj._consume_pending_prompt_selection("9")

        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert consumed is True
        assert "out of range" in printed.lower()
        assert cli_obj._pending_prompt_messages is None

    def test_pending_non_numeric_falls_through_and_disarms(self):
        cli_obj = _make_cli()
        cli_obj._pending_prompt_messages = [
            {"id": 22, "timestamp": None, "preview": "second prompt"},
        ]

        consumed = cli_obj._consume_pending_prompt_selection("hello there")

        assert consumed is False
        assert cli_obj._pending_prompt_messages is None

    def test_other_slash_command_disarms_pending_prompt_selection(self):
        cli_obj = _make_cli()
        cli_obj._pending_prompt_messages = [
            {"id": 22, "timestamp": None, "preview": "second prompt"},
        ]
        cli_obj.show_help = MagicMock()

        cli_obj.process_command("/help")

        assert cli_obj._pending_prompt_messages is None

    def test_prompt_text_flattens_multimodal_user_content(self):
        cli_obj = _make_cli()
        cli_obj._session_db.get_user_message.return_value = {
            "id": 11,
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "file://image.png"}},
                {"type": "text", "text": "world"},
            ],
        }

        assert cli_obj._prompt_text_for_message_id(11) == "hello\nworld"

    def test_prompts_without_session_db_reports_unavailable(self):
        cli_obj = _make_cli()
        cli_obj._session_db = None

        with patch("cli._cprint") as mock_cprint:
            cli_obj._handle_prompts_command("/prompts")

        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert "Session history database" in printed or "session" in printed.lower()
        assert cli_obj._pending_prompt_messages is None


class TestCliPromptsRealDb:
    """Real SessionDB integration tests for /prompts classic CLI."""

    def test_prompts_list_with_real_db(self, tmp_path, capsys):
        from hermes_state import SessionDB

        cli_obj = _make_cli()
        db = SessionDB(db_path=tmp_path / "state.db")
        cli_obj._session_db = db
        cli_obj.session_id = "s1"
        db.create_session("s1", "cli")
        db.append_message("s1", role="user", content="first prompt")
        db.append_message("s1", role="assistant", content="answer")
        db.append_message("s1", role="user", content="second prompt")

        cli_obj._handle_prompts_command("/prompts")
        output = capsys.readouterr().out

        assert "Recent prompts" in output
        assert "second prompt" in output
        assert "first prompt" in output
        assert cli_obj._pending_prompt_messages is not None
        assert len(cli_obj._pending_prompt_messages) == 2
        # Newest first
        assert cli_obj._pending_prompt_messages[0]["preview"] == "second prompt"
        db.close()

    def test_prompts_select_with_real_db(self, tmp_path):
        from hermes_state import SessionDB

        cli_obj = _make_cli()
        app = _App()
        cli_obj._app = app
        db = SessionDB(db_path=tmp_path / "state.db")
        cli_obj._session_db = db
        cli_obj.session_id = "s1"
        db.create_session("s1", "cli")
        db.append_message("s1", role="user", content="first prompt full text")
        db.append_message("s1", role="assistant", content="answer")
        db.append_message("s1", role="user", content="second prompt full text")

        cli_obj._handle_prompts_command("/prompts 2")

        # Index 2 = first prompt (newest is #1)
        assert app.current_buffer.text == "first prompt full text"
        assert cli_obj._pending_prompt_messages is None
        db.close()

    def test_prompts_no_session_id_with_real_db(self, tmp_path):
        from hermes_state import SessionDB

        cli_obj = _make_cli()
        db = SessionDB(db_path=tmp_path / "state.db")
        cli_obj._session_db = db
        cli_obj.session_id = None

        cli_obj._handle_prompts_command("/prompts")

        assert cli_obj._pending_prompt_messages is None
        db.close()
