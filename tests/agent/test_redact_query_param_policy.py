"""One place decides whether a URL query parameter carries a credential.

``_SENSITIVE_QUERY_PARAMS`` is the repository's answer to that question, and
`hermes dump` masks fallback-provider endpoints through it rather than keeping
a second name list of its own — a second list drifts, and the drift is silent
until a secret is already on a public paste. These tests pin the predicate that
makes the sharing possible.
"""

from agent.redact import is_sensitive_query_param


def test_listed_names_match_regardless_of_case_or_separator():
    for name in (
        "signature",
        "SIGNATURE",
        "x-amz-signature",
        "X-Amz-Signature",
        "code",
        "access_token",
        "access-token",
        "api_key",
        "client_secret",
    ):
        assert is_sensitive_query_param(name), name


def test_public_parameters_are_left_alone():
    # Masking these would cost diagnostics and buy no safety.
    for name in ("region", "model", "max_tokens", "version", "stream"):
        assert not is_sensitive_query_param(name), name


def test_non_string_names_do_not_raise():
    # Query names come from parsed config, which is whatever the user wrote.
    assert not is_sensitive_query_param(None)
    assert not is_sensitive_query_param(42)
