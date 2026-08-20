"""Tests for sanitize_anthropic_kwargs (#31673).

Guards the Anthropic Messages dispatch boundary against Responses-API-only
kwargs (``instructions``, ``input``, ``store``, ``parallel_tool_calls``)
leaking in under an api_mode-flip race. The Anthropic SDK raises a
non-retryable ``TypeError`` on any of them, killing the whole turn.
"""

import logging
from types import SimpleNamespace

import pytest

from agent.anthropic_adapter import (
    _OPENAI_ONLY_EXTRA_BODY_KEYS,
    _RESPONSES_ONLY_KWARGS,
    _client_base_url,
    create_anthropic_message,
    sanitize_anthropic_kwargs,
)


def _fake_anthropic_call(**kwargs):
    """Mimic the Anthropic SDK's strict kwarg signature."""
    allowed = {
        "model", "messages", "max_tokens", "system", "tools", "tool_choice",
        "extra_body", "extra_headers", "temperature", "top_p", "top_k",
        "thinking", "timeout",
    }
    bad = set(kwargs) - allowed
    if bad:
        raise TypeError(
            "Messages.stream() got an unexpected keyword argument "
            f"{sorted(bad)[0]!r}"
        )
    return "OK"


def test_bare_leaked_payload_reproduces_the_typeerror():
    """Without the guard, a Responses-shaped payload raises the issue's error."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _fake_anthropic_call(model="claude-sonnet-4-6", instructions="sys")


def test_strips_all_responses_only_keys():
    payload = {
        "model": "claude-sonnet-4-6",
        "instructions": "You are Hermes.",
        "input": [{"role": "user", "content": "hi"}],
        "store": False,
        "parallel_tool_calls": True,
    }
    out = sanitize_anthropic_kwargs(payload)
    assert out is payload  # mutates in place and returns same dict
    assert payload == {"model": "claude-sonnet-4-6"}
    assert _fake_anthropic_call(**payload) == "OK"




def test_warns_when_keys_are_stripped(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.anthropic_adapter"):
        sanitize_anthropic_kwargs(
            {"model": "m", "instructions": "sys"}, log_prefix="[pfx] "
        )
    assert any(
        "31673" in r.message and "[pfx] " in r.message
        for r in caplog.records
    ), caplog.records


class TestOpenAIOnlyExtraBodyStripping:
    """OpenAI chat-shape ``extra_body`` keys must not reach api.anthropic.com.

    These differ from the Responses-only kwargs above: the SDK accepts them
    locally and forwards ``extra_body`` verbatim, so the failure is a hard
    remote 400 ``<key>: Extra inputs are not permitted`` rather than a local
    TypeError. Verified live against api.anthropic.com for every key in
    ``_OPENAI_ONLY_EXTRA_BODY_KEYS``.

    Real-world symptom: session title generation passes
    ``extra_body={"response_format": {...json_schema...}}`` because it shares
    one call site across all providers; on an Anthropic-routed auxiliary task
    every title request 400s and sessions fall back to derived titles.
    """

    ANTHROPIC = "https://api.anthropic.com"

    def test_response_format_stripped_for_native_anthropic(self):
        payload = {
            "model": "claude-opus-5",
            "extra_body": {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "session_title"},
                }
            },
        }
        sanitize_anthropic_kwargs(payload, base_url=self.ANTHROPIC)
        assert "extra_body" not in payload, (
            "extra_body should be dropped entirely once its only key is "
            "removed -- an empty dict is pointless request weight"
        )

    def test_every_known_openai_only_key_is_stripped(self):
        for key in sorted(_OPENAI_ONLY_EXTRA_BODY_KEYS):
            payload = {"model": "m", "extra_body": {key: "x"}}
            sanitize_anthropic_kwargs(payload, base_url=self.ANTHROPIC)
            assert "extra_body" not in payload, f"{key} survived stripping"

    def test_anthropic_native_extra_body_keys_survive(self):
        """Only the OpenAI-shaped keys go; genuine passthrough is preserved."""
        payload = {
            "model": "m",
            "extra_body": {
                "response_format": {"type": "json_object"},
                "speed": "fast",
                "metadata": {"user_id": "abc"},
            },
        }
        sanitize_anthropic_kwargs(payload, base_url=self.ANTHROPIC)
        assert payload["extra_body"] == {
            "speed": "fast",
            "metadata": {"user_id": "abc"},
        }

    def test_third_party_gateways_keep_their_passthrough(self):
        """Anthropic-compatible proxies deliberately accept vendor fields."""
        for base_url in (
            "https://api.moonshot.ai/anthropic",
            "https://api.deepseek.com/anthropic",
            "https://bedrock-runtime.eu-west-2.amazonaws.com",
            "https://my-proxy.internal/v1",
        ):
            payload = {
                "model": "m",
                "extra_body": {"response_format": {"type": "json_object"}},
            }
            sanitize_anthropic_kwargs(payload, base_url=base_url)
            assert payload["extra_body"] == {
                "response_format": {"type": "json_object"}
            }, f"{base_url} lost its vendor passthrough"

    def test_omitted_base_url_defaults_to_native_anthropic(self):
        """No base_url means direct api.anthropic.com, so the guard applies.

        This matches ``_is_third_party_anthropic_endpoint``'s own documented
        convention ("No base_url = direct Anthropic API"). Defaulting the
        other way would leave the default call path -- the one that actually
        400s in production -- unprotected.
        """
        payload = {
            "model": "m",
            "extra_body": {"response_format": {"type": "json_object"}},
        }
        sanitize_anthropic_kwargs(payload)
        assert "extra_body" not in payload

    def test_non_dict_and_empty_extra_body_are_safe(self):
        for value in (None, {}, "not-a-dict", 42):
            payload = {"model": "m", "extra_body": value}
            sanitize_anthropic_kwargs(payload, base_url=self.ANTHROPIC)
        payload = {"model": "m"}
        sanitize_anthropic_kwargs(payload, base_url=self.ANTHROPIC)
        assert payload == {"model": "m"}

    def test_both_guards_apply_in_one_call(self):
        """Responses-only kwargs and OpenAI extra_body keys strip together."""
        payload = {
            "model": "m",
            "instructions": "sys",
            "extra_body": {"seed": 7, "speed": "fast"},
        }
        sanitize_anthropic_kwargs(payload, base_url=self.ANTHROPIC)
        assert payload == {"model": "m", "extra_body": {"speed": "fast"}}

    def test_parallel_tool_calls_covered_by_both_lists(self):
        """It is a top-level Responses kwarg AND an OpenAI body key."""
        assert "parallel_tool_calls" in _RESPONSES_ONLY_KWARGS
        assert "parallel_tool_calls" in _OPENAI_ONLY_EXTRA_BODY_KEYS
        payload = {
            "model": "m",
            "parallel_tool_calls": True,
            "extra_body": {"parallel_tool_calls": True},
        }
        sanitize_anthropic_kwargs(payload, base_url=self.ANTHROPIC)
        assert payload == {"model": "m"}


class TestClientBaseUrlHelper:
    def test_reads_base_url_from_client(self):
        client = SimpleNamespace(base_url="https://api.anthropic.com")
        assert _client_base_url(client) == "https://api.anthropic.com"

    def test_missing_or_raising_attribute_returns_none(self):
        assert _client_base_url(SimpleNamespace()) is None

        class Exploding:
            @property
            def base_url(self):
                raise RuntimeError("boom")

        assert _client_base_url(Exploding()) is None


class TestCreateAnthropicMessageIntegration:
    """The guard must be wired into the real dispatch path, not just callable."""

    def test_create_anthropic_message_strips_before_dispatch(self):
        seen = {}

        class FakeMessages:
            def stream(self, **kwargs):
                seen.update(kwargs)
                raise RuntimeError("stop-after-capture")

        client = SimpleNamespace(
            base_url="https://api.anthropic.com", messages=FakeMessages()
        )
        payload = {
            "model": "claude-opus-5",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"response_format": {"type": "json_object"}},
        }
        with pytest.raises(RuntimeError, match="stop-after-capture"):
            create_anthropic_message(client, payload)
        assert "extra_body" not in seen, (
            "create_anthropic_message must strip OpenAI-only extra_body keys "
            "before they reach the Anthropic SDK"
        )






