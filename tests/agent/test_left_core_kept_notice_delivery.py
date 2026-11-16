"""Notices a left-core migration kept for a home (a gateway start, a CLI resolution) reach the first turn
someone sees: an agent nobody watches (cron run, subagent, background fork) starts and runs without
claiming them, so they are not dropped on an agent that has no way to show them."""
from types import SimpleNamespace

import pytest

import hermes_cli.left_core_migration as lcm
from agent.agent_init import _load_tools
from agent.status_output import StatusOutputMixin
from agent.turn_context import _reset_per_turn_agent_state

KEPT = "✓ GitHub Copilot moved out of core — installed the 'github-copilot' plugin from the catalog."


class Agent(StatusOutputMixin):
    log_prefix = ""
    quiet_mode = save_trajectories = _use_prompt_caching = False
    ephemeral_system_prompt = _compression_warning = _memory_store = None
    status_callback = notice_callback = tool_progress_callback = None
    api_mode = "anthropic_messages"
    max_iterations = 1

    def __init__(self, platform, **attrs):
        self.platform, self.printed = platform, []
        self._tool_guardrails = SimpleNamespace(reset_for_turn=lambda: None)
        self._warning_presentation_enabled = lambda: True
        self.__dict__.update(attrs)

    def _vprint(self, message, **kwargs):
        self.printed.append(message)


@pytest.fixture
def home(tmp_path, monkeypatch):
    import model_tools
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(lcm, "_attempted", set())
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr(model_tools, "check_toolset_requirements", dict)
    lcm._queue_notice(tmp_path, KEPT)  # kept by the gateway's start
    return tmp_path


def _start_and_run_a_turn(agent):
    _load_tools(agent, None, None)
    _reset_per_turn_agent_state(agent)
    return agent


def test_unwatched_agents_leave_kept_notices_for_the_chat_agent(home):
    notices = []
    _start_and_run_a_turn(Agent("cron"))
    _start_and_run_a_turn(Agent("subagent", _delegate_depth=1, notice_callback=notices.append))
    _start_and_run_a_turn(Agent("telegram", _persist_disabled=True, suppress_status_output=True,
                                notice_callback=notices.append))  # a background-review fork
    assert notices == []
    chat = Agent("telegram")
    _load_tools(chat, None, None)
    chat.notice_callback = notices.append  # the gateway wires it per turn, after __init__
    _reset_per_turn_agent_state(chat)
    _reset_per_turn_agent_state(chat)
    assert [notice.text for notice in notices] == [KEPT]


def test_a_cli_turn_prints_what_another_process_kept(home):
    agent = _start_and_run_a_turn(Agent("cli"))
    assert agent.printed == [KEPT]
    assert lcm._claim_notices(home) == []
