from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import json

import pytest

from gateway.platform_context import (
    AuthenticatedPlatformContext,
    authenticated_platform_context_scope,
    get_authenticated_platform_context,
)
from model_tools import handle_function_call
from tools.registry import registry


_SCHEMA = {"description": "test", "parameters": {"type": "object"}}


@pytest.fixture
def registered_tool():
    name = "_test_authenticated_platform_context"
    registered = []

    def register(handler):
        registry.register(name, "test", _SCHEMA, handler, is_async=inspect.iscoroutinefunction(handler))
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
    with authenticated_platform_context_scope(trusted):
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


def test_forged_explicit_context_is_rejected_when_ambient_context_is_bound(registered_tool):
    trusted = AuthenticatedPlatformContext("telegram", "bot-1", "user-1", "chat-1")
    forged = AuthenticatedPlatformContext("telegram", "bot-1", "attacker", "chat-1")

    @registered_tool
    def handler(args, *, authenticated_platform_context):
        return json.dumps({"user_id": authenticated_platform_context.user_id})

    with authenticated_platform_context_scope(trusted):
        with pytest.raises(ValueError, match="cannot replace ambient"):
            handle_function_call(
                "_test_authenticated_platform_context", {},
                authenticated_platform_context=forged,
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True,
            )


def test_explicit_context_cannot_create_authority_without_ambient_context(registered_tool):
    forged = AuthenticatedPlatformContext("telegram", "bot-1", "attacker", "chat-1")

    @registered_tool
    def handler(args, *, authenticated_platform_context):
        return json.dumps({"user_id": authenticated_platform_context.user_id})

    with pytest.raises(ValueError, match="requires ambient"):
        handle_function_call(
            "_test_authenticated_platform_context", {},
            authenticated_platform_context=forged,
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )


def test_handlers_without_context_kwarg_still_work(registered_tool):
    @registered_tool
    def handler(args):
        return json.dumps({"value": args["value"]})

    trusted = AuthenticatedPlatformContext(
        platform="telegram", account_id="bot-1", user_id="user-1", chat_id="chat-1"
    )
    with authenticated_platform_context_scope(trusted):
        result = handle_function_call(
            "_test_authenticated_platform_context",
            {"value": "ok"},
            authenticated_platform_context=trusted,
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


def test_nested_code_cannot_replace_ambient_context():
    trusted = AuthenticatedPlatformContext("telegram", "bot-1", "user-1", "chat-1")
    forged = AuthenticatedPlatformContext("telegram", "bot-1", "attacker", "chat-1")
    with authenticated_platform_context_scope(trusted):
        with pytest.raises(ValueError, match="cannot replace ambient"):
            from gateway.platform_context import set_authenticated_platform_context
            set_authenticated_platform_context(forged)
        assert get_authenticated_platform_context() is trusted


def test_context_scope_resets_after_exception_and_does_not_use_model_args():
    trusted = AuthenticatedPlatformContext("telegram", "bot-1", "user-1", "chat-1")
    assert get_authenticated_platform_context() is None
    with pytest.raises(RuntimeError):
        with authenticated_platform_context_scope(trusted):
            assert get_authenticated_platform_context() is trusted
            raise RuntimeError("boom")
    assert get_authenticated_platform_context() is None


def test_direct_registry_and_executor_dispatch_receive_context(registered_tool):
    seen = []

    @registered_tool
    def handler(args, *, authenticated_platform_context):
        seen.append(authenticated_platform_context)
        return json.dumps({"ok": True})

    trusted = AuthenticatedPlatformContext("telegram", "bot-1", "user-1", "chat-1")
    with authenticated_platform_context_scope(trusted):
        assert json.loads(registry.dispatch("_test_authenticated_platform_context", {})) == {"ok": True}
        with ThreadPoolExecutor(max_workers=1) as pool:
            # The production executor wrapper uses copy_context; this assertion covers the
            # ContextVar contract independently of tool arguments.
            import contextvars
            assert pool.submit(contextvars.copy_context().run, registry.dispatch,
                              "_test_authenticated_platform_context", {}).result()
    assert seen == [trusted, trusted]


def test_async_registry_dispatch_receives_context(registered_tool):
    seen = []

    @registered_tool
    async def handler(args, *, authenticated_platform_context):
        seen.append(authenticated_platform_context)
        return json.dumps({"ok": True})

    trusted = AuthenticatedPlatformContext("telegram", "bot-1", "user-1", "chat-1")
    with authenticated_platform_context_scope(trusted):
        assert json.loads(registry.dispatch("_test_authenticated_platform_context", {})) == {"ok": True}
    assert seen == [trusted]


def test_tool_call_recursion_keeps_context_out_of_model_arguments(monkeypatch, registered_tool):
    seen = []

    @registered_tool
    def handler(args, *, authenticated_platform_context):
        seen.append((args, authenticated_platform_context))
        return json.dumps({"ok": True})

    import model_tools
    bridge_calls = 0

    def bridge(*args, **kwargs):
        nonlocal bridge_calls
        bridge_calls += 1
        return (None, ("_test_authenticated_platform_context", {"value": "model"})) if bridge_calls == 1 else None

    monkeypatch.setattr(
        model_tools,
        "_dispatch_bridge_tool",
        bridge,
    )
    trusted = AuthenticatedPlatformContext("telegram", "bot-1", "user-1", "chat-1")
    with authenticated_platform_context_scope(trusted):
        result = model_tools.handle_function_call("tool_call", {"calls": []})
    assert json.loads(result) == {"ok": True}
    assert seen == [({"value": "model"}, trusted)]


def test_nested_dispatch_cannot_replace_ambient_context(registered_tool):
    trusted = AuthenticatedPlatformContext("telegram", "bot-1", "user-1", "chat-1")
    forged = AuthenticatedPlatformContext("telegram", "bot-1", "attacker", "chat-1")
    seen = []

    @registered_tool
    def handler(args, *, authenticated_platform_context):
        seen.append(authenticated_platform_context)
        return json.dumps({"ok": True})

    with authenticated_platform_context_scope(trusted):
        with pytest.raises(ValueError, match="cannot replace ambient"):
            handle_function_call(
                "_test_authenticated_platform_context", {},
                authenticated_platform_context=forged,
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True,
            )
    assert seen == []
