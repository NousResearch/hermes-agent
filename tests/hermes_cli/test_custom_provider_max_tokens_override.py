"""Custom-provider ``max_tokens`` reaches ``request_overrides``.

A ``max_tokens`` (or ``max_output_tokens`` alias) key on a ``providers:`` custom
entry must be lifted into the resolved runtime's ``request_overrides`` — that is
the documented path by which per-provider output caps reach the request, so a
relay-side default cap cannot truncate long scheduled replies.
"""

from hermes_cli.runtime_provider_custom import _custom_provider_request_overrides


class TestCustomProviderMaxTokensOverride:
    def test_max_tokens_lifted_into_request_overrides(self):
        provider = {
            "name": "grid",
            "base_url": "https://grid.example/v1",
            "max_tokens": 8192,
        }
        overrides = _custom_provider_request_overrides(provider)
        assert overrides.get("max_tokens") == 8192

    def test_max_output_tokens_alias_lifted(self):
        provider = {
            "name": "grid",
            "base_url": "https://grid.example/v1",
            "max_output_tokens": 4096,
        }
        overrides = _custom_provider_request_overrides(provider)
        assert overrides.get("max_tokens") == 4096

    def test_explicit_max_tokens_wins_over_alias(self):
        provider = {
            "name": "grid",
            "base_url": "https://grid.example/v1",
            "max_tokens": 8192,
            "max_output_tokens": 4096,
        }
        overrides = _custom_provider_request_overrides(provider)
        assert overrides.get("max_tokens") == 8192

    def test_non_positive_values_ignored(self):
        provider = {
            "name": "grid",
            "base_url": "https://grid.example/v1",
            "max_tokens": 0,
        }
        assert _custom_provider_request_overrides(provider) == {}

    def test_non_int_values_ignored(self):
        provider = {
            "name": "grid",
            "base_url": "https://grid.example/v1",
            "max_tokens": "8192",
        }
        assert _custom_provider_request_overrides(profile := provider) == {}

    def test_extra_body_still_forwarded_alongside_max_tokens(self):
        provider = {
            "name": "grid",
            "base_url": "https://grid.example/v1",
            "max_tokens": 8192,
            "extra_body": {"provider": {"order": ["gpu"]}},
        }
        overrides = _custom_provider_request_overrides(provider)
        assert overrides["extra_body"] == {"provider": {"order": ["gpu"]}}
        assert overrides["max_tokens"] == 8192

    def test_empty_entry_returns_no_overrides(self):
        assert (
            _custom_provider_request_overrides({
                "name": "grid",
                "base_url": "https://grid.example/v1",
            })
            == {}
        )
