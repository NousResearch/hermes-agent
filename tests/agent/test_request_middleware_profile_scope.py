"""Request middleware follows the AIAgent's immutable profile ownership."""

from pathlib import Path
from unittest.mock import patch


_PLUGIN = """
def register(ctx):
    def stamp(**kwargs):
        return {
            "request": {**kwargs["request"], "profile_middleware": "secondary"},
            "source": "secondary",
        }
    ctx.register_middleware("llm_request", stamp)
"""


def _write_profile_plugin(home: Path) -> None:
    plugin = home / "plugins" / "owner-middleware"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: owner-middleware\n"
        "version: 0.1.0\n"
        "provides_middleware:\n"
        "  - llm_request\n",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - owner-middleware\n",
        encoding="utf-8",
    )


def test_request_middleware_uses_agent_profile_after_ambient_scope_changes(tmp_path):
    from agent.turn_api_request import build_api_request
    from agent.turn_context import _PER_TURN_RESET_STATE
    from hermes_cli.plugins import discover_plugins
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from run_agent import AIAgent

    launch_home = get_hermes_home()
    secondary_home = tmp_path / "profiles" / "secondary"
    secondary_home.mkdir(parents=True)
    (secondary_home / "sessions").mkdir()
    _write_profile_plugin(secondary_home)

    token = set_hermes_home_override(str(secondary_home))
    try:
        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
        ):
            agent = AIAgent(
                provider="custom",
                model="test-model",
                base_url="http://127.0.0.1:1/v1",
                api_key="sk-test",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_id="profile-owned-request",
            )
        discover_plugins()
    finally:
        reset_hermes_home_override(token)

    assert launch_home != secondary_home
    for name, value in _PER_TURN_RESET_STATE:
        setattr(agent, name, value)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    try:
        built = build_api_request(
            agent,
            api_messages=messages,
            _moa_prepared_request=None,
            tools_for_api=agent.tools,
            system_message="system",
            messages=messages,
            original_user_message="user",
            approx_tokens=0,
            total_chars=0,
            retry_count=0,
            api_call_count=1,
            api_request_id="request-1",
            api_start_time=0.0,
            effective_task_id="task-1",
            turn_id="turn-1",
        )
    finally:
        agent.close()

    assert built.api_kwargs["profile_middleware"] == "secondary"
    assert built._llm_middleware_trace == [{"source": "secondary"}]
