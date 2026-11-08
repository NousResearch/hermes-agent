"""The Telegram system prompt steers fixed-choice decisions to ``clarify`` (native inline buttons).

Without it the model faked decision controls with Telegram copy-to-clipboard buttons, which never
send a choice back to the waiting session.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt_parts


def test_telegram_stable_prompt_prefers_clarify_buttons_over_fake_controls():
    agent = SimpleNamespace(
        platform="telegram", load_soul_identity=False, skip_context_files=False, valid_tool_names=[],
        _task_completion_guidance=False, _tool_use_enforcement=False, _environment_probe=False,
        _kanban_worker_guidance="", _memory_store=None, _memory_manager=None, _platform_hint_overrides={},
        model="", provider="", pass_session_id=False, session_id="")
    with (patch("agent.prompt_builder.load_soul_md", return_value=""),
          patch("agent.prompt_builder.build_environment_hints", return_value=""),
          patch("agent.prompt_builder.build_context_files_prompt", return_value="")):
        stable = build_system_prompt_parts(agent)["stable"]
    assert "You are on Telegram." in stable
    assert "call `clarify`" in stable and "tappable buttons" in stable
    assert "copy-to-clipboard" in stable
