from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace


def test_contextual_max_iteration_summary_bypasses_relay(monkeypatch):
    from agent import chat_completion_helpers, relay_llm

    calls = []

    class _Completions:
        @staticmethod
        def create(**kwargs):
            calls.append(kwargs)
            return object()

    class _Client:
        chat = SimpleNamespace(completions=_Completions())

    class _Transport:
        @staticmethod
        def normalize_response(_response):
            return SimpleNamespace(content="direct contextual summary")

    agent = SimpleNamespace(
        max_iterations=3,
        provider="openai",
        model="gpt-test",
        api_mode="chat_completions",
        _contextual_execution=True,
        _cached_system_prompt="",
        ephemeral_system_prompt=None,
        prefill_messages=[],
        max_tokens=None,
        reasoning_config=None,
        base_url="https://provider.invalid/v1",
        _base_url_lower="https://provider.invalid/v1",
        openrouter_min_coding_score=None,
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        provider_price=None,
        provider_require_parameters=False,
        provider_data_collection=None,
        session_id="session-1",
        _safe_print=lambda *_args, **_kwargs: None,
        _should_sanitize_tool_calls=lambda: False,
        _copy_reasoning_content_for_api=lambda _source, _target: None,
        _sanitize_api_messages=lambda messages: messages,
        _drop_thinking_only_and_merge_users=lambda messages: messages,
        _supports_reasoning_extra_body=lambda: False,
        _is_openrouter_url=lambda: False,
        _ensure_primary_openai_client=lambda **_kwargs: _Client(),
        _get_transport=lambda: _Transport(),
    )

    monkeypatch.setattr(
        relay_llm,
        "execute_current",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("contextual summary must not enter Relay")
        ),
    )
    monkeypatch.setattr(relay_llm, "complete_logical_call", lambda *_a, **_k: None)

    result = chat_completion_helpers.handle_max_iterations(
        agent,
        [{"role": "user", "content": "continue"}],
        api_call_count=3,
    )

    assert result == "direct contextual summary"
    assert len(calls) == 1


def test_every_compression_site_fails_closed_for_contextual_execution():
    from agent import conversation_loop

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(conversation_loop.run_conversation))
    )
    guarded_calls = 0

    for node in ast.walk(tree):
        for _field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index, statement in enumerate(value):
                if not isinstance(statement, ast.Assign):
                    continue
                calls = [
                    call
                    for call in ast.walk(statement.value)
                    if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "_compress_context"
                ]
                if not calls:
                    continue
                assert index > 0
                guard = value[index - 1]
                assert isinstance(guard, ast.If)
                assert isinstance(guard.test, ast.Name)
                assert guard.test.id == "_contextual_execution"
                assert any(isinstance(item, ast.Return) for item in guard.body)
                guarded_calls += len(calls)

    assert guarded_calls == 6
