"""extra_body.cache_prompt must not survive onto media-carrying turns (#108659).

llama-server style backends reuse the prompt/KV cache slot keyed on the textual
prefix, so two different images sent back-to-back in one session can be served
the first request's answer. The fix forces ``extra_body.cache_prompt`` off for
the one call that carries media and leaves every plain-text turn untouched.
"""

from __future__ import annotations

import pytest

from agent.chat_completion_helpers import (
    _disable_cache_prompt_for_media,
    _has_media_content,
    build_api_kwargs,
)
from run_agent import AIAgent

_IMAGE_PART = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
_MEDIA_MSGS = [
    {
        "role": "user",
        "content": [
            _IMAGE_PART,
            {"type": "text", "text": "what color is this?"},
        ],
    }
]
_TEXT_MSGS = [{"role": "user", "content": "what color is this?"}]


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
        {"type": "input_video", "video_url": "data:video/mp4;base64,AAAA"},
        {"type": "video", "source": {"type": "base64", "media_type": "video/mp4", "data": "AAAA"}},
    ],
)
def test_detects_media_part_on_every_wire_shape(part):
    assert _has_media_content([{"role": "user", "content": [part]}]) is True


@pytest.mark.parametrize(
    "content",
    [
        "plain text",
        [{"type": "text", "text": "hello"}],
        None,
    ],
)
def test_text_only_payload_is_not_media(content):
    assert _has_media_content([{"role": "user", "content": content}]) is False


def test_empty_or_non_message_payloads_are_not_media():
    assert _has_media_content([]) is False
    assert _has_media_content(None) is False
    assert _has_media_content(["not-a-dict"]) is False


def test_media_call_forces_cache_prompt_off_without_mutating_the_caller():
    overrides = {"extra_body": {"cache_prompt": True, "other": 1}}
    out = _disable_cache_prompt_for_media(overrides, _MEDIA_MSGS)
    assert out["extra_body"]["cache_prompt"] is False
    assert out["extra_body"]["other"] == 1
    assert overrides["extra_body"]["cache_prompt"] is True
    assert overrides is not out


def test_text_call_keeps_the_configured_cache_prompt():
    overrides = {"extra_body": {"cache_prompt": True}}
    assert _disable_cache_prompt_for_media(overrides, _TEXT_MSGS) is overrides


def test_absent_or_disabled_key_is_a_noop():
    media = _MEDIA_MSGS
    assert _disable_cache_prompt_for_media({}, media) == {}
    off = {"extra_body": {"cache_prompt": False}}
    assert _disable_cache_prompt_for_media(off, media) is off
    assert _disable_cache_prompt_for_media(None, media) is None


# Dummy credential for a localhost-only test backend; not a real secret.
_LOCAL_BACKEND_KEY = "test-" + "key"


def _custom_vision_agent():
    return AIAgent(
        api_key=_LOCAL_BACKEND_KEY,
        base_url="http://127.0.0.1:8080/v1",
        model="qwen-vl-test",
        provider="custom",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id="sess-cache-prompt-108659",
    )


def test_media_turn_builds_kwargs_with_cache_prompt_off():
    agent = _custom_vision_agent()
    agent.request_overrides = {"extra_body": {"cache_prompt": True}}
    kwargs = build_api_kwargs(agent, _MEDIA_MSGS)
    assert kwargs["extra_body"]["cache_prompt"] is False


def test_text_turn_keeps_cache_prompt_at_the_call_site():
    agent = _custom_vision_agent()
    agent.request_overrides = {"extra_body": {"cache_prompt": True}}
    kwargs = build_api_kwargs(agent, _TEXT_MSGS)
    assert kwargs["extra_body"]["cache_prompt"] is True
