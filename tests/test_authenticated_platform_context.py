from __future__ import annotations

import json

import pytest

from gateway.platform_context import AuthenticatedPlatformContext
from model_tools import handle_function_call
from tools.registry import registry


_SCHEMA = {"description": "test", "parameters": {"type": "object"}}


@pytest.fixture
def registered_tool():
    name = "_test_authenticated_platform_context"
    registered = []

    def register(handler):
        registry.register(name, "test", _SCHEMA, handler)
        registered.append(name)
        return handler

    yield register
    for tool_name in registered:
        registry.deregister(tool_name)


def test_context_is_internal_and_model_args_cannot_override_it(registered_tool):
    seen = {}

    @registered_tool
    def handler(args, *, authenticated_platform_context):
        seen["args"] = args
        seen["context"] = authenticated_platform_context
        return json.dumps({"ok": True})

    trusted = AuthenticatedPlatformContext(
        platform="telegram", account_id="bot-1", user_id="user-1", chat_id="chat-1"
    )
    result = handle_function_call(
        "_test_authenticated_platform_context",
        {"authenticated_platform_context": {"platform": "forged"}},
        authenticated_platform_context=trusted,
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    assert json.loads(result) == {"ok": True}
    assert seen["args"]["authenticated_platform_context"]["platform"] == "forged"
    assert seen["context"] is trusted


def test_handlers_without_context_kwarg_still_work(registered_tool):
    @registered_tool
    def handler(args):
        return json.dumps({"value": args["value"]})

    result = handle_function_call(
        "_test_authenticated_platform_context",
        {"value": "ok"},
        authenticated_platform_context=AuthenticatedPlatformContext(
            platform="telegram", account_id="bot-1", user_id="user-1", chat_id="chat-1"
        ),
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    assert json.loads(result) == {"value": "ok"}


def test_context_is_immutable():
    context = AuthenticatedPlatformContext(
        platform="telegram", account_id="bot-1", user_id="user-1", chat_id="chat-1"
    )

    with pytest.raises(AttributeError):
        context.user_id = "forged"
