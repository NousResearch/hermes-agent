"""The "model not found" row of ``_PROVIDER_ERROR_REPLIES`` (#124162 review).

A misconfigured model id on a self-hosted endpoint answers 400 with a model-not-found envelope.
The generic fallback ("kept failing — use /retry") points at the one action that cannot fix it,
so the row must win the shapes it is for and lose to the rows whose fix is different (rate-limit:
wait; auth: /login; policy: rephrase).
"""

import pytest

from gateway.run import _gateway_provider_error_reply, _GATEWAY_MODEL_NOT_FOUND_RE

NOT_FOUND_REPLY = "That model isn't available from the model service."


@pytest.mark.parametrize("text", [
    # llama.cpp's 400 envelope, verbatim.
    '{"error":{"message":"model \'deepseek-v3\' not found","type":"invalid_request_error","code":400}}',
    'Error: model "llama-3.1-8b" not found',
    "model not found",                                    # llama.cpp /v1/models, no name given
    "model x not found",
    '{"error":{"message":"model \\"strata-small\\" not found","type":"provider_error"}}',
    "The model 'gpt-4o' does not exist for model type: GENERATION",   # vLLM
    "The model `gpt-4o` does not exist or is not available for the organization.",  # OpenAI SDK
])
def test_model_not_found_names_the_fix_that_works(text):
    reply = _gateway_provider_error_reply(text)
    assert NOT_FOUND_REPLY in reply, reply
    assert "/model" in reply, reply
    # Telling the user to retry a request the endpoint will never accept is the bug being fixed.
    assert "/retry" not in reply, reply


@pytest.mark.parametrize("text, expected", [
    # Ordered table: rate-limit first, then auth, then policy — each has a different fix.
    ("429 rate limit exceeded; model 'x' not found", "rate-limiting"),
    ("HTTP 401: Unauthorized - model 'x' not found", "Sign-in"),
    ("Your request was blocked by safety policy for model 'x' not found", "rejected this request"),
])
def test_the_rows_with_a_different_fix_still_win(text, expected):
    assert expected in _gateway_provider_error_reply(text)


@pytest.mark.parametrize("text", [
    "the file does not exist",
    "tool 'x' does not exist",
    "model directory does not exist",       # a path, not a model id
    "no model found",
    "the model is loading",
    "upstream 502 bad gateway",
])
def test_it_does_not_claim_text_it_is_not_about(text):
    assert NOT_FOUND_REPLY not in _gateway_provider_error_reply(text)
    assert not _GATEWAY_MODEL_NOT_FOUND_RE.search(text)


def test_the_unquoted_does_not_exist_form_is_a_known_gap():
    """"does not exist" is claimed only in its quoted forms. Requiring quotes is what keeps
    "the file does not exist" and "model directory does not exist" unclaimed, and the price is
    this shape. Pinned as a gap so widening the pattern later is a decision, not a surprise."""
    assert not _GATEWAY_MODEL_NOT_FOUND_RE.search("The model gpt-4o does not exist")
