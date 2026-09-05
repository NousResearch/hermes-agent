"""CLI turn path must honor the per-provider output cap.

Contract: ``_ensure_runtime_credentials`` applies the resolved runtime's
``max_output_tokens`` onto ``self.max_tokens`` exactly like gateway/run.py
does — as a fallback only. The documented global keys (HERMES_MAX_TOKENS
env var, ``model.max_tokens`` in config.yaml) always win when set.

Regression context: the CLI unpacked the resolved runtime dict but
dropped the cap field, so `hermes chat` kept sending the generic custom
profile's default_max_tokens (65536) while the gateway honored the
configured per-provider cap.
"""

import pytest


class FakeHermesCLI:
    """Minimal stand-in exposing just what _ensure_runtime_credentials
    touches before the cap application — enough to unit-test the new
    behavior without constructing a full HermesCLI."""

    def __init__(self, *, max_tokens=None):
        self.max_tokens = max_tokens
        self.provider = None
        self.api_mode = "chat_completions"
        self.acp_command = None
        self.acp_args = []
        self._credential_pool = None
        self._provider_source = None
        self.api_key = "k"
        self.base_url = "https://relay.example.invalid/v1"
        self.model = "lumo-max"
        self.requested_provider = "aperture"


@pytest.fixture()
def apply_cap():
    """Import the real code path under test once it lands.

    The extraction target is the cap-application block inside
    HermesCLI._ensure_runtime_credentials; tests exercise its semantics
    through the same runtime-dict shape resolve_runtime_provider returns.
    """
    from hermes_cli.cli_agent_setup_mixin import (
        apply_per_provider_output_cap,
    )

    return apply_per_provider_output_cap


class TestApplyPerProviderOutputCap:
    """Mirror of gateway/run.py resolution order."""

    def test_applies_when_unset(self, apply_cap):
        cli = FakeHermesCLI(max_tokens=None)
        apply_cap(cli, {"max_output_tokens": 16384})
        assert cli.max_tokens == 16384

    def test_global_config_wins(self, apply_cap):
        cli = FakeHermesCLI(max_tokens=32768)
        apply_cap(cli, {"max_output_tokens": 16384})
        assert cli.max_tokens == 32768

    def test_missing_cap_is_noop(self, apply_cap):
        cli = FakeHermesCLI(max_tokens=None)
        apply_cap(cli, {})
        assert cli.max_tokens is None

    def test_invalid_cap_values_ignored(self, apply_cap):
        for bad in (0, -1, "16384", 16384.0, None):
            cli = FakeHermesCLI(max_tokens=None)
            apply_cap(cli, {"max_output_tokens": bad})
            assert cli.max_tokens is None, repr(bad)

    def test_malformed_global_falls_through_to_cap(self, apply_cap):
        """A zero/negative global max_tokens must not suppress the cap."""
        cli = FakeHermesCLI(max_tokens=-1)
        apply_cap(cli, {"max_output_tokens": 16384})
        assert cli.max_tokens == 16384

    def test_accepts_max_tokens_alias(self, apply_cap):
        cli = FakeHermesCLI(max_tokens=None)
        apply_cap(cli, {"max_tokens": 4096})
        assert cli.max_tokens == 4096
