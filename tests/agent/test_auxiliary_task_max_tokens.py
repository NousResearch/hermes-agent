"""Task-level output caps reach auxiliary calls and their recovery paths."""

from types import SimpleNamespace

import pytest

from agent import auxiliary_client
from agent.auxiliary_task_config import resolve_task_max_tokens, session_auxiliary_scope


@pytest.mark.parametrize("async_mode", [False, True])
def test_task_cap_reaches_primary_retry_and_fallback(monkeypatch, async_mode):
    observed = []

    def prepare(task, **kwargs):
        observed.append(kwargs["max_tokens"])
        return SimpleNamespace(
            effective_timeout=30.0, effective_extra_body={},
            resolved_base_url="https://openrouter.ai/api/v1", resolved_api_key="",
            resolved_api_mode="chat_completions", final_model="deepseek/deepseek-v4-flash",
        )

    monkeypatch.setattr(auxiliary_client, "_prepare_aux_request", prepare)
    messages = [{"role": "user", "content": "describe image"}]
    with session_auxiliary_scope({"vision": {"max_tokens": 3000}}):
        _, retry, fallback = auxiliary_client._plan_aux_call(
            "vision", async_mode=async_mode, provider="openrouter", model="deepseek/deepseek-v4-flash",
            base_url="https://openrouter.ai/api/v1", api_key=None, main_runtime={},
            messages=messages, temperature=None, max_tokens=None, tools=None, timeout=None,
            extra_body=None, reasoning_config=None, extra_headers=None, api_mode=None,
            route_info=None,
        )

    assert observed == [3000]
    assert retry["max_tokens"] == fallback["max_tokens"] == 3000
    wire = auxiliary_client._build_call_kwargs(
        "openrouter", "deepseek/deepseek-v4-flash", messages, max_tokens=retry["max_tokens"],
        base_url="https://openrouter.ai/api/v1", task="vision",
    )
    assert wire["max_tokens"] == 3000


def test_explicit_cap_overrides_task_config(monkeypatch):
    observed = []

    def prepare(task, **kwargs):
        observed.append(kwargs["max_tokens"])
        return SimpleNamespace(
            effective_timeout=30.0, effective_extra_body={}, resolved_base_url="",
            resolved_api_key="", resolved_api_mode="chat_completions", final_model="m",
        )

    monkeypatch.setattr(auxiliary_client, "_prepare_aux_request", prepare)
    with session_auxiliary_scope({"vision": {"max_tokens": 3000}}):
        auxiliary_client._plan_aux_call(
            "vision", async_mode=False, provider="openrouter", model="m", base_url=None,
            api_key=None, main_runtime={}, messages=[], temperature=None, max_tokens=700,
            tools=None, timeout=None, extra_body=None, reasoning_config=None,
            extra_headers=None, api_mode=None, route_info=None,
        )
    assert observed == [700]


def test_invalid_task_caps_are_ignored():
    for value in (0, -1, True, 1.5, "invalid"):
        with session_auxiliary_scope({"vision": {"max_tokens": value}}):
            assert resolve_task_max_tokens("vision", None) is None
