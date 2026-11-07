"""Regression: ``request_overrides["extra_body"]`` merges per-key on BOTH build paths.

Legacy (unregistered-provider) path built ``extra_body`` (reasoning payload, provider
routing, engine extensions) and then blindly ``api_kwargs.update(request_overrides)``,
so a caller passing ``extra_body={"top_k": 40}`` silently discarded every built
extension. The profile path already merged ``extra_body`` per key; both paths must
stay equivalent: merge under ``extra_body``, apply everything else top-level, same-key
collisions go to the caller, explicit zeros survive, inputs are never mutated.
"""

import json

import httpx
import pytest
from openai import OpenAI

from agent.transports import get_transport
from providers.base import ProviderProfile

MESSAGES = [{"role": "user", "content": "hello"}]


@pytest.fixture
def transport():
    return get_transport("chat_completions")


def _wire_body(kwargs):
    """Run transport kwargs through the real OpenAI SDK over a fixture transport.

    Returns the JSON body the SDK serialized onto the wire — proof the SDK accepted
    the kwargs and folded ``extra_body`` entries into the top-level request body.
    """
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chatcmpl_merge",
            "object": "chat.completion",
            "created": 0,
            "model": kwargs["model"],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAI(
            api_key="test-key", base_url="https://merge-capable.test/v1", http_client=http_client,
        )
        client.chat.completions.create(**kwargs)
    return captured


# Reasoning payload + engine extensions the legacy path builds before overrides ride in.
_LEGACY_PARAMS = dict(
    supports_reasoning=True,
    reasoning_config={"enabled": True, "effort": "high"},
    is_openrouter=True,
    provider_preferences={"only": ["groq"]},
    extra_body_additions={"repeat_penalty": 1.1, "seed": 0},
)


class TestLegacyPathExtraBodyMerge:
    """No provider_profile → the flag-built legacy kwargs path."""

    def _build(self, transport, request_overrides, **params):
        merged = dict(_LEGACY_PARAMS)
        merged.update(params)
        return transport.build_kwargs(model="merge-model", messages=MESSAGES, tools=None,
                                      request_overrides=request_overrides, **merged)

    def test_extra_body_override_keeps_built_reasoning_and_provider_extensions(self, transport):
        kwargs = self._build(transport, {"extra_body": {"top_k": 40, "min_p": 0.0}})
        body = kwargs["extra_body"]

        assert body["reasoning"] == {"enabled": True, "effort": "high"}
        assert body["provider"] == {"only": ["groq"]}
        assert body["top_k"] == 40
        assert body["min_p"] == 0.0

    def test_explicit_zero_values_survive_merge(self, transport):
        # seed=0 rides in the built additions, min_p=0.0 in the caller's overrides —
        # neither may be dropped for being falsy.
        kwargs = self._build(transport, {"extra_body": {"top_k": 40, "min_p": 0.0}})
        body = kwargs["extra_body"]

        assert body["seed"] == 0
        assert body["min_p"] == 0.0

    def test_same_key_collision_goes_to_caller(self, transport):
        kwargs = self._build(transport, {"extra_body": {"repeat_penalty": 1.3}})
        body = kwargs["extra_body"]

        assert body["repeat_penalty"] == 1.3  # caller beats the built value
        assert body["seed"] == 0              # untouched keys stay
        assert body["reasoning"]["effort"] == "high"

    def test_top_level_overrides_keep_applying(self, transport):
        # Non-extra_body overrides must still land top-level with caller precedence.
        kwargs = self._build(
            transport, {"temperature": 0.2, "max_tokens": 512, "extra_body": {"top_k": 40}},
        )

        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 512
        assert kwargs["extra_body"]["top_k"] == 40
        assert kwargs["extra_body"]["reasoning"] == {"enabled": True, "effort": "high"}

    def test_non_dict_extra_body_override_cannot_wipe_built_extensions(self, transport):
        # Parity with the profile path: a malformed (non-dict) extra_body override may
        # not delete the built extensions when any exist.
        kwargs = self._build(transport, {"extra_body": "not-a-dict"})
        body = kwargs["extra_body"]

        assert isinstance(body, dict)
        assert body["reasoning"] == {"enabled": True, "effort": "high"}

    def test_inputs_are_not_mutated(self, transport):
        additions = {"repeat_penalty": 1.1, "seed": 0}
        overrides = {"extra_body": {"top_k": 40}}

        kwargs = self._build(transport, overrides, extra_body_additions=additions)
        kwargs["extra_body"]["canary"] = True

        assert overrides == {"extra_body": {"top_k": 40}}
        assert additions == {"repeat_penalty": 1.1, "seed": 0}
        assert "canary" not in overrides["extra_body"]
        assert "canary" not in additions

    def test_override_with_no_built_extra_body_still_lands(self, transport):
        # Pin: nothing built + caller extra_body → that extra_body is sent as-is.
        kwargs = transport.build_kwargs(
            model="merge-model", messages=MESSAGES, tools=None,
            request_overrides={"extra_body": {"top_k": 40}},
        )

        assert kwargs["extra_body"] == {"top_k": 40}

    def test_no_request_overrides_leaves_built_extra_body_intact(self, transport):
        kwargs = self._build(transport, None)
        body = kwargs["extra_body"]

        assert body["reasoning"] == {"enabled": True, "effort": "high"}
        assert body["provider"] == {"only": ["groq"]}
        assert body["repeat_penalty"] == 1.1


class TestProfilePathExtraBodyMerge:
    """ProviderProfile path already merges per key — pinned so both paths stay equivalent."""

    def _build(self, transport, request_overrides):
        profile = ProviderProfile(name="merge-test")
        return transport.build_kwargs(
            model="merge-model", messages=MESSAGES, tools=None, provider_profile=profile,
            extra_body_additions={"repeat_penalty": 1.1, "seed": 0},
            request_overrides=request_overrides,
        )

    def test_profile_path_merges_extra_body_per_key(self, transport):
        kwargs = self._build(transport, {"extra_body": {"top_k": 40, "min_p": 0.0, "repeat_penalty": 1.3}})
        body = kwargs["extra_body"]

        assert body == {"repeat_penalty": 1.3, "seed": 0, "top_k": 40, "min_p": 0.0}

    def test_profile_path_top_level_override_still_applies(self, transport):
        kwargs = self._build(transport, {"temperature": 0.2, "extra_body": {"top_k": 40}})

        assert kwargs["temperature"] == 0.2
        assert kwargs["extra_body"]["seed"] == 0


class TestSdkWireSerialization:
    """The actual OpenAI SDK must accept the merged kwargs and serialize engine
    extensions onto the wire — fixture transport, no live inference."""

    def test_sdk_serializes_top_k_and_min_p_alongside_built_extensions(self, transport):
        kwargs = transport.build_kwargs(
            model="merge-model", messages=MESSAGES, tools=None, **_LEGACY_PARAMS,
            request_overrides={"extra_body": {"top_k": 40, "min_p": 0.0}},
        )
        body = _wire_body(kwargs)

        assert body["top_k"] == 40
        assert body["min_p"] == 0.0
        assert body["seed"] == 0
        assert body["reasoning"] == {"enabled": True, "effort": "high"}
        assert body["provider"] == {"only": ["groq"]}
        assert "extra_body" not in body  # SDK folds it into the top level, never nests it
