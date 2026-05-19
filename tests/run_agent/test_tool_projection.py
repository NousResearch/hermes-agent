import copy
import json
from types import SimpleNamespace

from agent.tool_projection import (
    INTENT_KEY,
    PatchToolSurface,
    project_messages_for_patch_surface,
    projected_valid_tool_names,
    refresh_agent_tool_projection,
    resolve_patch_tool_surface,
    responses_tools_for_surface,
)


def _patch_tool():
    return {
        "type": "function",
        "function": {
            "name": "patch",
            "description": "Hermes patch",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _read_tool():
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_codex_responses_gets_freeform_apply_patch_surface():
    tools = [_patch_tool(), _read_tool()]

    surface = resolve_patch_tool_surface(
        tools,
        api_mode="codex_responses",
        provider="openai-codex",
        model="gpt-5.3-codex",
    )

    assert surface == PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH
    assert projected_valid_tool_names(tools, patch_surface=surface) == {
        "apply_patch",
        "read_file",
    }


def test_unknown_responses_provider_keeps_hermes_patch_surface():
    tools = [_patch_tool(), _read_tool()]

    surface = resolve_patch_tool_surface(
        tools,
        api_mode="codex_responses",
        provider="custom",
        base_url="https://example.invalid/v1",
        model="gpt-5.3",
    )

    assert surface == PatchToolSurface.HERMES_PATCH
    assert projected_valid_tool_names(tools, patch_surface=surface) == {
        "patch",
        "read_file",
    }


def test_frozen_agent_projection_survives_fallback_provider_mutation():
    agent = SimpleNamespace(
        tools=[_patch_tool(), _read_tool()],
        api_mode="codex_responses",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-5.4-mini",
    )

    refresh_agent_tool_projection(agent, freeze=True)
    assert agent._patch_tool_surface == PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH
    assert agent.valid_tool_names == {"apply_patch", "read_file"}

    agent.provider = "openai"
    agent.base_url = "https://api.openai.com/v1"
    agent.model = "gpt-5.4-mini"
    refresh_agent_tool_projection(agent)

    assert agent._patch_tool_surface == PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH
    assert agent.valid_tool_names == {"apply_patch", "read_file"}


def test_responses_schema_uses_custom_freeform_tool_not_hosted_apply_patch():
    converted = responses_tools_for_surface(
        [_patch_tool(), _read_tool()],
        patch_surface=PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH,
    )

    assert {"type": "apply_patch"} not in converted
    assert [tool.get("name") for tool in converted if tool["type"] == "function"] == ["read_file"]
    custom = [tool for tool in converted if tool["type"] == "custom"]
    assert len(custom) == 1
    assert custom[0]["name"] == "apply_patch"
    assert custom[0]["format"]["syntax"] == "lark"


def test_patch_mode_history_projects_to_apply_patch_without_mutating_storage():
    patch = "*** Begin Patch\n*** Delete File: old.txt\n*** End Patch\n"
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "patch",
                        "arguments": json.dumps({"mode": "patch", "patch": patch}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "patch",
            "tool_call_id": "call_1",
            "content": '{"success": true}',
        },
    ]
    stored = copy.deepcopy(messages)

    projected = project_messages_for_patch_surface(
        messages,
        patch_surface=PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH,
    )

    assert messages == stored
    tc = projected[0]["tool_calls"][0]
    assert tc["type"] == "apply_patch"
    assert tc["function"]["name"] == "apply_patch"
    assert json.loads(tc["function"]["arguments"]) == {"patch": patch}
    assert projected[1]["name"] == "apply_patch"


def test_apply_patch_history_projects_to_hermes_patch_and_strips_private_intent():
    original_args = {
        "mode": "replace",
        "path": "src/app.py",
        "old_string": "foo",
        "new_string": "bar",
        "replace_all": True,
    }
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "apply_patch",
                    "function": {
                        "name": "apply_patch",
                        "arguments": json.dumps({"patch": "ignored when intent exists"}),
                    },
                    INTENT_KEY: {
                        "canonical_tool": "patch",
                        "canonical_arguments": original_args,
                        "lowered_to": "apply_patch",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "apply_patch",
            "tool_call_id": "call_2",
            "content": '{"success": true}',
        },
    ]

    projected = project_messages_for_patch_surface(
        messages,
        patch_surface=PatchToolSurface.HERMES_PATCH,
    )

    tc = projected[0]["tool_calls"][0]
    assert INTENT_KEY not in tc
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "patch"
    assert json.loads(tc["function"]["arguments"]) == original_args
    assert projected[1]["name"] == "patch"
    assert INTENT_KEY in messages[0]["tool_calls"][0]
