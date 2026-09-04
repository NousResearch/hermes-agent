"""End-to-end boundaries for generic tool exception secret redaction."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import model_tools
from agent.redact import TOOL_SECRET_PLACEHOLDER
from agent.secret_scope import reset_secret_scope, set_secret_scope
from run_agent import AIAgent
from tools.registry import registry


def _tool_defs(name: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "synthetic local boundary tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _target_tool() -> str:
    excluded = {
        "todo",
        "memory",
        "session_search",
        "delegate_task",
        "execute_code",
        "tool_search",
        "tool_call",
        "tool_describe",
    }
    return next(name for name in registry.get_all_tool_names() if name not in excluded)


def _canary() -> str:
    return "stage1b-session-" + ("R4kN" * 12)


@pytest.fixture()
def agent(tmp_path):
    target = _target_tool()
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs(target)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        instance = AIAgent(
            api_key="test-" + ("local" * 4),
            base_url="https://example.invalid/v1",
            quiet_mode=False,
            verbose_logging=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance.client = MagicMock()
    instance.logs_dir = tmp_path
    instance.session_id = "stage1b-session"
    instance.session_start = datetime.now()
    instance._session_json_enabled = True
    instance._session_db = MagicMock()
    instance._session_db.append_messages_batch.side_effect = (
        lambda **kwargs: list(range(1, len(kwargs["messages"]) + 1))
    )
    instance._session_db_created = True
    instance._last_flushed_db_idx = 0
    instance._flushed_db_message_ids = set()
    instance._flushed_db_message_session_id = None
    instance._persist_disabled = False
    return instance


def test_generic_exception_is_redacted_before_model_ui_and_transcript(
    agent, tmp_path, capsys, caplog
):
    target = _target_tool()
    entry = registry.get_entry(target)
    original_handler, original_async = entry.handler, entry.is_async
    canary = _canary()
    scope_token = set_secret_scope({"STAGE1B_RUNTIME_SECRET": canary})
    progress_results = []
    complete_results = []
    agent.tool_progress_callback = (
        lambda event, _name, _preview, _args, **kwargs:
        progress_results.append(kwargs.get("result"))
        if event == "tool.completed"
        else None
    )
    agent.tool_complete_callback = (
        lambda _call_id, _name, _args, result: complete_results.append(result)
    )

    def boom(_args, **_kwargs):
        raise RuntimeError(canary + " permission denied")

    entry.handler, entry.is_async = boom, False
    call_id = "stage1b-call"
    assistant_message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(name=target, arguments="{}"),
            )
        ]
    )
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": target, "arguments": "{}"},
                }
            ],
        }
    ]

    try:
        with caplog.at_level("DEBUG"):
            agent._execute_tool_calls_sequential(
                assistant_message,
                messages,
                "stage1b-task",
            )
        agent._save_session_log(messages)
    finally:
        entry.handler, entry.is_async = original_handler, original_async
        reset_secret_scope(scope_token)

    user_visible = capsys.readouterr().out
    model_tool_content = messages[-1]["content"]
    snapshot_path = tmp_path / "session_stage1b-session.json"
    transcript = snapshot_path.read_text(encoding="utf-8")
    persisted_batches = repr(agent._session_db.append_messages_batch.call_args_list)
    callback_payloads = repr(progress_results + complete_results)
    all_surfaces = (
        user_visible
        + model_tool_content
        + transcript
        + persisted_batches
        + callback_payloads
        + caplog.text
    )

    assert canary not in all_surfaces
    assert TOOL_SECRET_PLACEHOLDER in model_tool_content
    assert TOOL_SECRET_PLACEHOLDER in transcript
    assert TOOL_SECRET_PLACEHOLDER in persisted_batches
    assert TOOL_SECRET_PLACEHOLDER in callback_payloads
    assert "RuntimeError" in caplog.text
    assert "permission denied" in all_surfaces
    assert json.loads(model_tool_content)["error"].startswith("[TOOL_ERROR]")


def test_transform_hook_cannot_reintroduce_runtime_secret(monkeypatch):
    target = _target_tool()
    entry = registry.get_entry(target)
    original_handler, original_async = entry.handler, entry.is_async
    canary = _canary()
    scope_token = set_secret_scope({"STAGE1B_RUNTIME_SECRET": canary})

    def clean_result(_args, **_kwargs):
        return json.dumps({"ok": True})

    def invoke_hook(name, **_kwargs):
        if name == "transform_tool_result":
            return [json.dumps({"error": canary})]
        return []

    entry.handler, entry.is_async = clean_result, False
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "transform_tool_result")
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke_hook)
    try:
        out = model_tools.handle_function_call(
            target,
            {},
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )
    finally:
        entry.handler, entry.is_async = original_handler, original_async
        reset_secret_scope(scope_token)

    assert canary not in out
    assert TOOL_SECRET_PLACEHOLDER in out


def test_execution_middleware_result_is_redacted_before_observer(monkeypatch):
    target = _target_tool()
    canary = _canary()
    scope_token = set_secret_scope({"STAGE1B_RUNTIME_SECRET": canary})
    observed = []

    monkeypatch.setattr(
        "hermes_cli.middleware.run_tool_execution_middleware",
        lambda *_args, **_kwargs: canary + " permission denied",
    )
    monkeypatch.setattr(
        model_tools,
        "_emit_post_tool_call_hook",
        lambda **kwargs: observed.append(kwargs["result"]),
    )
    try:
        out = model_tools.handle_function_call(
            target,
            {},
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
        )
    finally:
        reset_secret_scope(scope_token)

    assert canary not in out
    assert canary not in repr(observed)
    assert TOOL_SECRET_PLACEHOLDER in out
    assert TOOL_SECRET_PLACEHOLDER in repr(observed)
    assert "permission denied" in out
