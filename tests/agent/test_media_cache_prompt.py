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
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
        },
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
        {"type": "input_video", "video_url": "data:video/mp4;base64,AAAA"},
        {
            "type": "video",
            "source": {"type": "base64", "media_type": "video/mp4", "data": "AAAA"},
        },
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


_BASE_URL = "http://127.0.0.1:8080/v1"


class TestAuxiliaryMediaCachePrompt:
    """All three auxiliary request paths must receive a sanitized extra_body copy.

    The guard lives in ``_build_call_kwargs`` — the single boundary shared by the
    initial aux call (incl. the vision task), the same-provider credential retry,
    and provider fallback (#108659).
    """

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # Minimal config so task/provider resolution reads a clean home
        (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")

    def test_build_call_kwargs_boundary_sanitizes_media_turn(self):
        from agent.auxiliary_client import _build_call_kwargs

        caller_extra = {"cache_prompt": True, "keep": 1}
        kwargs = _build_call_kwargs(
            "custom",
            "qwen-vl-test",
            _MEDIA_MSGS,
            extra_body=caller_extra,
            base_url=_BASE_URL,
        )
        assert kwargs["extra_body"]["cache_prompt"] is False
        assert kwargs["extra_body"]["keep"] == 1
        assert caller_extra["cache_prompt"] is True  # caller's dict untouched

    def test_build_call_kwargs_boundary_keeps_text_turn(self):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            "custom",
            "qwen-vl-test",
            _TEXT_MSGS,
            extra_body={"cache_prompt": True},
            base_url=_BASE_URL,
        )
        assert kwargs["extra_body"]["cache_prompt"] is True

    def test_initial_vision_call_receives_sanitized_copy(self):
        from agent.auxiliary_client import _prepare_aux_request

        call_extra = {"cache_prompt": True}
        prepared = _prepare_aux_request(
            "vision",
            provider="custom",
            model="qwen-vl-test",
            base_url=_BASE_URL,
            api_key=_LOCAL_BACKEND_KEY,
            main_runtime={},
            messages=_MEDIA_MSGS,
            temperature=None,
            max_tokens=None,
            tools=None,
            timeout=None,
            extra_body=call_extra,
            reasoning_config=None,
            extra_headers=None,
            api_mode=None,
            route_info=None,
            async_mode=False,
        )
        assert prepared.kwargs["extra_body"]["cache_prompt"] is False
        assert call_extra["cache_prompt"] is True  # call-site override dict untouched

    def test_same_provider_credential_retry_receives_sanitized_copy(self):
        from agent.auxiliary_client import _prepare_same_provider_retry

        effective_extra = {"cache_prompt": True}
        _client, retry_kwargs = _prepare_same_provider_retry(
            task="vision",
            resolved_provider="custom",
            resolved_model="qwen-vl-test",
            resolved_base_url=_BASE_URL,
            resolved_api_key=_LOCAL_BACKEND_KEY,
            resolved_api_mode=None,
            main_runtime={},
            final_model="qwen-vl-test",
            messages=_MEDIA_MSGS,
            temperature=None,
            max_tokens=None,
            tools=None,
            effective_timeout=5.0,
            effective_extra_body=effective_extra,
            reasoning_config=None,
            async_mode=False,
        )
        assert retry_kwargs["extra_body"]["cache_prompt"] is False
        assert effective_extra["cache_prompt"] is True

    def test_provider_fallback_receives_sanitized_copy(self):
        from agent.auxiliary_client import (
            _FallbackDestination,
            _fallback_request_kwargs,
        )

        destination = _FallbackDestination(
            provider="custom", base_url=_BASE_URL, api_mode=None, model="qwen-vl-test"
        )
        effective_extra = {"cache_prompt": True}
        fb_kwargs = _fallback_request_kwargs(
            destination,
            task="vision",
            messages=_MEDIA_MSGS,
            tools=None,
            temperature=None,
            max_tokens=None,
            effective_timeout=5.0,
            effective_extra_body=effective_extra,
            reasoning_config=None,
            fallback_entry={},
            task_config={},
            apply_fast_lane=False,
        )
        assert fb_kwargs["extra_body"]["cache_prompt"] is False
        assert effective_extra["cache_prompt"] is True


class TestCustomProviderConfigChain:
    """custom_providers.<name>.extra_body.cache_prompt flows through the real config chain.

    Loads a temporary HERMES_HOME config.yaml via the real resolver/precedence chain
    (no direct ``request_overrides`` assignment) and checks the end-to-end behavior:
    media forces the key off, text keeps the configured value, and the loaded
    config object is never mutated.
    """

    @pytest.fixture(autouse=True)
    def _hermes_home(self, tmp_path, monkeypatch):
        import yaml

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config = {
            "model": {"default": "qwen-vl-test", "provider": "custom"},
            "custom_providers": [
                {
                    "name": "llama-local",
                    "base_url": _BASE_URL,
                    "api_key": _LOCAL_BACKEND_KEY,
                    "extra_body": {"cache_prompt": True},
                }
            ],
        }
        (hermes_home / "config.yaml").write_text(yaml.dump(config))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def test_media_forces_off_text_keeps_configured_value(self):
        agent = _custom_vision_agent()
        loaded = agent.request_overrides.get("extra_body")
        # Prove the key really arrived from config.yaml through the resolver chain
        assert isinstance(loaded, dict) and loaded["cache_prompt"] is True

        media_kwargs = build_api_kwargs(agent, _MEDIA_MSGS)
        assert media_kwargs["extra_body"]["cache_prompt"] is False
        text_kwargs = build_api_kwargs(agent, _TEXT_MSGS)
        assert text_kwargs["extra_body"]["cache_prompt"] is True

        # The loaded config object is not mutated by either call
        assert agent.request_overrides["extra_body"] is loaded
        assert loaded["cache_prompt"] is True
