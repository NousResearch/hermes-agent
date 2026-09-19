"""Bedrock Converse reasoning-replay regressions for Kimi K3 (issue #115865).

Two invariants, both proven red on f0d8efe4e7d:

1. ``reasoningContent`` is a botocore *tagged union* — exactly one of ``reasoningText`` /
   ``redactedContent``. Replaying captured thinking under a bare ``text`` key dies client-side
   with ``ParamValidationError`` before the request leaves the process, so the assertion is
   made by botocore's own Converse input shape rather than a hand-written dict comparison.

2. Region-sealed encrypted reasoning must not be replayed against a model/region that did not
   mint it. Bedrock answers such a replay with ``ValidationException`` ("Encrypted content
   cannot be used in a different region from the one that created it."), and the corrupted
   payload otherwise survives an in-place model switch into the Claude fallback. The live
   agent loop calls ``chat_completion_helpers._bedrock_converse_call`` (NOT the unused
   ``bedrock_adapter.call_converse_stream`` helper), so the self-heal is pinned on that path.
"""

from unittest.mock import MagicMock

import pytest

CROSS_REGION_REJECTION = (
    "An error occurred (ValidationException) when calling the ConverseStream operation: "
    "The model returned the following errors: "
    '{"error":{"code":"validation_error","message":"Encrypted content cannot be used in a '
    'different region from the one that created it.","param":null,'
    '"type":"invalid_request_error"}}'
)


def _converse_input_shape():
    botocore_session = pytest.importorskip("botocore.session")
    return botocore_session.get_session().get_service_model(
        "bedrock-runtime"
    ).operation_model("Converse").input_shape


def test_replayed_thinking_satisfies_the_converse_tagged_union():
    """Captured Kimi K3 thinking replays as a shape botocore accepts (#115865, symptom 1).

    Validated against the real Converse input shape: a bare ``text`` key raises
    ``ParamValidationError`` ("must be one of: reasoningText, redactedContent"), and packing
    ``reasoningText`` + ``redactedContent`` into one block trips the tagged-union arity check.
    """
    from botocore.validate import validate_parameters

    from agent.bedrock_adapter import _replay_ordered_blocks

    blocks = _replay_ordered_blocks([
        {"reasoningContent": {"text": "let me think", "redactedContentBase64": "cjE="}},
        {"toolUse": {"toolUseId": "call_1", "name": "read_file", "input": {}}},
    ])

    validate_parameters(
        {"modelId": "global.moonshotai.kimi-k3",
         "messages": [{"role": "assistant", "content": blocks}]},
        _converse_input_shape(),
    )
    # The thinking text must survive the reshape, not be dropped to dodge the validator.
    assert any(
        block.get("reasoningContent", {}).get("reasoningText", {}).get("text") == "let me think"
        for block in blocks
    )
    # Order is load-bearing: reasoning precedes the toolUse it justifies.
    assert [key for block in blocks for key in block] == [
        "reasoningContent", "reasoningContent", "toolUse",
    ]


def test_cross_region_encrypted_reasoning_is_dropped_and_resent_on_the_live_path():
    """A region-sealed reasoning blob is stripped and resent once (#115865, symptom 2).

    Pinned on ``_bedrock_converse_call`` — the seam the agent loop actually streams through —
    so the retry cannot silently regress into the unused ``call_converse_stream`` helper.
    """
    from agent.bedrock_adapter import _bedrock_runtime_client_cache, reset_client_cache
    from agent.chat_completion_helpers import _bedrock_converse_call

    reset_client_cache()
    ok_response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "done"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1},
    }
    client = MagicMock()
    client.converse.side_effect = [Exception(CROSS_REGION_REJECTION), ok_response]
    _bedrock_runtime_client_cache["us-east-1"] = client

    api_kwargs = {
        "__bedrock_region__": "us-east-1",
        "__bedrock_converse__": True,
        "modelId": "global.moonshotai.kimi-k3",
        "messages": [
            {"role": "user", "content": [{"text": "go"}]},
            {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "let me think"}}},
                {"reasoningContent": {"redactedContent": b"region-sealed"}},
                {"toolUse": {"toolUseId": "call_1", "name": "read_file", "input": {}}},
            ]},
            {"role": "user", "content": [
                {"toolResult": {"toolUseId": "call_1", "content": [{"text": "ok"}]}},
            ]},
        ],
    }

    try:
        response = _bedrock_converse_call(dict(api_kwargs), stream=False)
    finally:
        reset_client_cache()

    assert response.choices[0].message.content == "done"
    assert client.converse.call_count == 2, "the sealed blob must be stripped and resent once"
    resent = client.converse.call_args_list[1].kwargs["messages"]
    assistant = next(m for m in resent if m["role"] == "assistant")
    assert not any(
        "redactedContent" in block.get("reasoningContent", {}) for block in assistant["content"]
    ), "region-sealed encrypted reasoning must not be replayed"
    # Unsealed reasoning text and the toolUse it precedes are preserved, so the turn stays
    # coherent instead of degrading into a bare tool call.
    assert [key for block in assistant["content"] for key in block] == [
        "reasoningContent", "toolUse",
    ]


def test_a_reasoning_only_turn_is_dropped_without_breaking_role_alternation():
    """Stripping a turn that held nothing but a sealed blob keeps the payload replayable.

    Converse rejects both an empty content list and two consecutive same-role turns, so the
    drop has to merge the neighbours it exposes rather than leave ``user, user`` behind.
    """
    from botocore.validate import validate_parameters

    from agent.bedrock_adapter_reasoning_replay import strip_sealed_reasoning

    kwargs = {
        "modelId": "global.moonshotai.kimi-k3",
        "messages": [
            {"role": "user", "content": [{"text": "go"}]},
            {"role": "assistant", "content": [
                {"reasoningContent": {"redactedContent": b"region-sealed"}},
            ]},
            {"role": "user", "content": [{"text": "still there?"}]},
        ],
    }

    stripped = strip_sealed_reasoning(kwargs)

    assert stripped is not kwargs, "a sealed blob must be detected as strippable"
    validate_parameters(stripped, _converse_input_shape())
    assert [m["role"] for m in stripped["messages"]] == ["user"]
    assert [b.get("text") for b in stripped["messages"][0]["content"]] == ["go", "still there?"]
    # Identity contract: nothing left to strip means a resend cannot help.
    assert strip_sealed_reasoning(stripped) is stripped
