"""Regression tests for the opt-in local qwen3.5 compact system prompt."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.system_prompt import (
    _use_local_compact_prompt,
    build_system_prompt,
    build_system_prompt_parts,
)
from agent.turn_context import build_api_messages
from hermes_cli.config_defaults import DEFAULT_CONFIG


def _gate_agent(**overrides):
    values = {
        "_local_compact_prompt": True,
        "model": "qwen3.5:9b-64k",
        "_base_url_hostname": "127.0.0.1",
        "valid_tool_names": {"terminal"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("hostname", ["localhost", "127.0.0.1", "::1", "LOCALHOST."])
def test_compact_prompt_activates_for_supported_loopback_hosts(hostname):
    assert _use_local_compact_prompt(_gate_agent(_base_url_hostname=hostname))


def test_compact_prompt_is_disabled_by_default():
    assert DEFAULT_CONFIG["agent"]["local_compact_prompt"] is False
    assert not _use_local_compact_prompt(_gate_agent(_local_compact_prompt=False))
    assert not _use_local_compact_prompt(_gate_agent(_local_compact_prompt="true"))


@pytest.mark.parametrize(
    ("model", "hostname", "tools"),
    [
        ("qwen3.5:9b-64k", "api.b.ai", {"terminal"}),
        ("qwen3.5:9b-64k", "192.168.1.10", {"terminal"}),
        ("qwen3.5:9b-64k", "0.0.0.0", {"terminal"}),
        ("qwen3.5:9b-64k", "ollama.example.com", {"terminal"}),
        ("qwen3:8b", "127.0.0.1", {"terminal"}),
        ("qwen3.5:9b-64k", "127.0.0.1", set()),
    ],
)
def test_compact_prompt_fails_closed_outside_narrow_scope(model, hostname, tools):
    assert not _use_local_compact_prompt(
        _gate_agent(model=model, _base_url_hostname=hostname, valid_tool_names=tools)
    )


def test_compact_prompt_keeps_execution_discipline_and_caller_prompt():
    parts = build_system_prompt_parts(
        _gate_agent(), system_message="PROFILE LOCAL INSTRUCTION"
    )
    assert "Hermes Agent" in parts["stable"]
    assert "tool" in parts["stable"].lower()
    assert "fabricat" in parts["stable"].lower()
    assert parts["context"] == "PROFILE LOCAL INSTRUCTION"
    assert parts["volatile"] == ""


def test_profile_ephemeral_prompt_is_still_appended_at_api_time():
    agent = _gate_agent(ephemeral_system_prompt="EPHEMERAL PROFILE INSTRUCTION")
    agent._copy_reasoning_content_for_api = lambda *_args: None
    agent._should_sanitize_tool_calls = lambda: False
    system_prompt = build_system_prompt(agent)
    messages, effective = build_api_messages(
        agent,
        [{"role": "user", "content": "do the task"}],
        current_turn_user_idx=0,
        ext_prefetch_cache=None,
        plugin_user_context=None,
        moa_config=None,
        active_system_prompt=system_prompt,
    )
    assert effective == system_prompt + "\n\nEPHEMERAL PROFILE INSTRUCTION"
    assert messages[0] == {"role": "system", "content": effective}


def _normal_agent(**overrides):
    values = {
        "load_soul_identity": False,
        "skip_context_files": True,
        "valid_tool_names": {"terminal"},
        "_task_completion_guidance": True,
        "_parallel_tool_call_guidance": True,
        "_tool_use_enforcement": "auto",
        "_execution_guidance": "auto",
        "_environment_probe": False,
        "_bot_mode_protocol": False,
        "_kanban_worker_guidance": "",
        "_memory_enabled": False,
        "_user_profile_enabled": False,
        "_memory_store": None,
        "_memory_manager": None,
        "_local_compact_prompt": False,
        "_base_url_hostname": "api.b.ai",
        "model": "qwen3.5:9b-64k",
        "provider": "bai",
        "platform": "cli",
        "pass_session_id": False,
        "session_id": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _normal_parts(agent):
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.system_prompt._coding_parts", return_value=([], [], [])),
        patch("agent.system_prompt._post_workspace_parts", return_value=[]),
        patch("agent.system_prompt._skills_prompt", return_value=""),
        patch("agent.system_prompt._frozen_plugin_prompt_sections", return_value=()),
        patch("agent.system_prompt._timestamp_line", return_value="STAMP"),
    ):
        return build_system_prompt_parts(agent)


def test_bai_cloud_request_uses_original_prompt_even_when_opted_in():
    baseline = _normal_parts(_normal_agent())
    opted_in = _normal_parts(_normal_agent(_local_compact_prompt=True))
    assert opted_in == baseline
    assert "STAMP" in opted_in["volatile"]


def test_bai_cloud_serialized_messages_are_identical_when_opted_in():
    def wire_request(agent):
        agent.ephemeral_system_prompt = ""
        agent._copy_reasoning_content_for_api = lambda *_args: None
        agent._should_sanitize_tool_calls = lambda: False
        prompt = "\n\n".join(filter(None, _normal_parts(agent).values()))
        return build_api_messages(
            agent,
            [{"role": "user", "content": "do the task"}],
            current_turn_user_idx=0,
            ext_prefetch_cache=None,
            plugin_user_context=None,
            moa_config=None,
            active_system_prompt=prompt,
        )

    assert wire_request(_normal_agent()) == wire_request(
        _normal_agent(_local_compact_prompt=True)
    )


def test_inactive_feature_leaves_normal_prompt_assembly_byte_identical():
    without_attribute = _normal_agent()
    del without_attribute._local_compact_prompt
    explicit_false = _normal_agent(_local_compact_prompt=False)
    assert _normal_parts(without_attribute) == _normal_parts(explicit_false)
