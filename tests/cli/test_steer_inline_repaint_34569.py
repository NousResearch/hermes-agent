"""Behavioral regression guard for issue #34569.

Inline /model and /steer commands must clear the prompt buffer and invalidate
prompt_toolkit immediately. This calls the extracted topic method through the
public facade and does not inspect source text.
"""
from types import SimpleNamespace

from hermes_cli.cli_tui_mixin import CLITuiMixin


def test_inline_command_reset_branches_invalidate():
    for command in ("/model test", "/steer test"):
        cli = CLITuiMixin()
        calls = []
        setattr(cli, "_should_handle_model_command_inline", lambda text, has_images: text.startswith("/model"))
        setattr(cli, "_should_handle_steer_command_inline", lambda text, has_images: text.startswith("/steer"))
        setattr(cli, "_should_handle_background_command_inline", lambda text, has_images: False)
        setattr(cli, "process_command", lambda text: calls.append(text) or True)
        cli._should_exit = False
        buffer_obj = SimpleNamespace(reset=lambda **kw: calls.append(("reset", kw)))
        event = SimpleNamespace(
            app=SimpleNamespace(
                current_buffer=buffer_obj,
                is_running=True,
                invalidate=lambda: calls.append("invalidate"),
            )
        )

        assert cli._tui_enter_inline_command(event, command, has_images=False) is True
        assert calls == [command, ("reset", {"append_to_history": True}), "invalidate"]
