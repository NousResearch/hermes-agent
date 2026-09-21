"""Custom provider request-overrides must carry output caps to the request path.

Issue #118066: a per-provider ``max_tokens`` (or ``max_output_tokens``) key is
silently dropped by ``_custom_provider_request_overrides`` — only ``extra_body``
is lifted — so cron/custom-path requests go out with ``max_tokens=None`` and the
endpoint applies its own default output cap, truncating replies.
"""

import hermes_cli.runtime_provider_custom as rpc


def test_extra_body_still_lifted():
    overrides = rpc._custom_provider_request_overrides({"extra_body": {"temperature": 0.2}})
    assert overrides == {"extra_body": {"temperature": 0.2}}


def test_max_tokens_lifted_into_request_overrides():
    overrides = rpc._custom_provider_request_overrides({"max_tokens": 8192})
    assert overrides.get("max_tokens") == 8192


def test_max_output_tokens_alias_lifted():
    overrides = rpc._custom_provider_request_overrides({"max_output_tokens": 4096})
    assert overrides.get("max_tokens") == 4096


def test_non_positive_or_non_int_max_tokens_ignored():
    assert "max_tokens" not in rpc._custom_provider_request_overrides({"max_tokens": 0})
    assert "max_tokens" not in rpc._custom_provider_request_overrides({"max_tokens": -5})
    assert "max_tokens" not in rpc._custom_provider_request_overrides({"max_tokens": "auto"})
    assert "max_tokens" not in rpc._custom_provider_request_overrides({"max_tokens": True})


def test_empty_entry_returns_empty():
    assert rpc._custom_provider_request_overrides({}) == {}


def test_max_tokens_and_extra_body_coexist():
    overrides = rpc._custom_provider_request_overrides(
        {"max_tokens": 8192, "extra_body": {"top_p": 0.9}}
    )
    assert overrides.get("max_tokens") == 8192
    assert overrides.get("extra_body") == {"top_p": 0.9}
