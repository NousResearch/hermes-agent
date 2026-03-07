import os
import re
from types import SimpleNamespace
from unittest.mock import MagicMock


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def test_render_markdown_response_box_formats_markdown():
    from cli import _render_markdown_response_box

    rendered = _render_markdown_response_box(
        "# Plan\n\nUse `rg` and **pytest**.\n\n- first\n- second\n\n```bash\necho hi\n```",
        width=80,
    )
    plain = _strip_ansi(rendered)

    assert "⚕ Hermes" in plain
    assert "Plan" in plain
    assert "rg" in plain
    assert "pytest" in plain
    assert "echo hi" in plain
    assert "# Plan" not in plain
    assert "`rg`" not in plain
    assert "**pytest**" not in plain
    assert "```bash" not in plain


def test_render_markdown_response_box_disables_hyperlink_escape_sequences():
    from cli import _render_markdown_response_box

    rendered = _render_markdown_response_box(
        "[Google](https://google.com)\n\n![Alt text](image-url.jpg)",
        width=80,
    )

    assert "\x1b]8;" not in rendered


def test_chat_routes_final_response_through_markdown_renderer(monkeypatch):
    from cli import HermesCLI

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.conversation_history = []
    cli_obj.session_id = "test-session"
    cli_obj.agent = SimpleNamespace(
        run_conversation=lambda **kwargs: {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "**done**"},
            ],
            "final_response": "**done**",
        },
        interrupt=MagicMock(),
    )
    cli_obj._ensure_runtime_credentials = lambda: True
    cli_obj._init_agent = lambda: True
    cli_obj._build_multimodal_content = lambda message, images: message
    cli_obj._print_assistant_response = MagicMock()

    monkeypatch.setattr("cli._cprint", lambda text: None)
    monkeypatch.setattr("cli.shutil.get_terminal_size", lambda: os.terminal_size((80, 24)))
    monkeypatch.setattr("cli.sys.stdout.flush", lambda: None)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    response = HermesCLI.chat(cli_obj, "hello")

    assert response == "**done**"
    cli_obj._print_assistant_response.assert_called_once_with("**done**")
