"""llama.cpp ``exceed_context_size_error`` parsing (tools/server/server-context.cpp).

The server's real window sits behind a paren after the measured request size; overflow recovery
must adopt it (never the request count) so compression converges on the served ``-c``.
"""

import pytest

from agent.model_metadata import get_context_length_from_provider_error, parse_context_limit_from_error

_REQUEST_MSG = "request (33056 tokens) exceeds the available context size (32768 tokens), try increasing it"


@pytest.mark.parametrize("msg", [
    _REQUEST_MSG,
    # The full stringified JSON body a client may surface (server-task.cpp::to_json).
    '{"error":{"code":400,"message":"' + _REQUEST_MSG + '","type":"exceed_context_size_error",'
    '"n_prompt_tokens":33056,"n_ctx":32768}}',
    # Same error type from the non-splittable batch path.
    "input (40000 tokens) is larger than the max context size (32768 tokens). skipping",
])
def test_llamacpp_exceed_context_size_error_yields_n_ctx(msg):
    assert parse_context_limit_from_error(msg) == 32768


def test_llamacpp_limit_is_adopted_only_when_lower():
    # Catalog guessed 131072 for a server enforcing 32768: adopt the reported limit.
    assert get_context_length_from_provider_error(_REQUEST_MSG, 131072) == 32768
    # Never raise the window from an error message.
    assert get_context_length_from_provider_error(_REQUEST_MSG, 16384) is None
