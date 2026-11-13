"""The Telegram system prompt steers fixed-choice decisions to ``clarify`` (native inline buttons).

Without it the model faked decision controls with Telegram copy-to-clipboard buttons, which never
send a choice back to the waiting session. The advice is only given when the session's toolset has
``clarify``: cron jobs delivering to Telegram and clarify-disabled profiles must not be told to call it.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.system_prompt import build_system_prompt_parts


def _stable(platform, valid_tool_names):
    agent = SimpleNamespace(
        platform=platform, load_soul_identity=False, skip_context_files=False, valid_tool_names=valid_tool_names,
        _task_completion_guidance=False, _tool_use_enforcement=False, _environment_probe=False,
        _kanban_worker_guidance="", _memory_store=None, _memory_manager=None, _platform_hint_overrides={},
        model="", provider="", pass_session_id=False, session_id="")
    with (patch("agent.prompt_builder.load_soul_md", return_value=""),
          patch("agent.prompt_builder.build_environment_hints", return_value=""),
          patch("agent.prompt_builder.build_context_files_prompt", return_value="")):
        return build_system_prompt_parts(agent)["stable"]


def test_telegram_stable_prompt_prefers_clarify_buttons_over_fake_controls():
    stable = _stable("telegram", ["clarify", "terminal"])
    assert "You are on Telegram." in stable
    assert "call `clarify`" in stable and "tappable buttons" in stable
    assert "copy-to-clipboard" in stable


def _cron_tool_names():
    from hermes_agent.cron.scheduler import _resolve_cron_disabled_toolsets
    from model_tools import get_tool_definitions
    defs = get_tool_definitions(disabled_toolsets=_resolve_cron_disabled_toolsets({}), quiet_mode=True)
    return [d["function"]["name"] for d in defs]


@pytest.mark.parametrize("surface", ["clarify_disabled_profile", "cron_delivering_to_telegram"])
def test_telegram_hint_never_recommends_clarify_without_the_tool(surface, monkeypatch):
    if surface == "cron_delivering_to_telegram":
        monkeypatch.setenv("HERMES_CRON_AUTO_DELIVER_PLATFORM", "telegram")
        names = _cron_tool_names()
        assert "clarify" not in names
        stable = _stable("cron", names)
        assert "Delivery destination (telegram): You are on Telegram." in stable
    else:
        stable = _stable("telegram", ["terminal", "read_file"])
        assert "You are on Telegram." in stable
    assert "clarify`" not in stable
    assert "copy-to-clipboard" not in stable
