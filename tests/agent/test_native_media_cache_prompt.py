from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as helpers


def _request_overrides_for(messages, monkeypatch):
    captured = {}
    overrides = {"extra_body": {"cache_prompt": True, "keep": "value"}}
    agent = SimpleNamespace(api_mode="chat_completions", tools=[])

    monkeypatch.setattr(helpers, "_reasoning_config_for_wire", lambda _agent: None)
    monkeypatch.setattr(helpers, "effective_request_overrides", lambda _agent: overrides)
    monkeypatch.setattr(helpers, "_prompt_cache_scope_for_agent", lambda _agent: "scope")
    monkeypatch.setattr(
        helpers,
        "_build_chat_completions_kwargs",
        lambda _agent, _messages, _tools, _reasoning, request_overrides, _scope: captured.update(
            request_overrides=request_overrides
        )
        or captured,
    )

    result = helpers._build_api_kwargs_for_mode(agent, messages)
    return overrides, result["request_overrides"]


@pytest.mark.parametrize("part_type", ["image_url", "input_image", "video_url", "input_video"])
def test_native_media_disables_configured_prompt_reuse_per_request(part_type, monkeypatch):
    configured, effective = _request_overrides_for(
        [{"role": "user", "content": [{"type": part_type, "url": "media"}]}], monkeypatch
    )

    assert effective == {"extra_body": {"cache_prompt": False, "keep": "value"}}
    assert configured["extra_body"]["cache_prompt"] is True


def test_text_request_preserves_configured_prompt_reuse(monkeypatch):
    configured, effective = _request_overrides_for(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}], monkeypatch
    )

    assert effective is configured
    assert effective["extra_body"]["cache_prompt"] is True