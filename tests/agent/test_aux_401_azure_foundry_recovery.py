"""Invariant tests: a 401 from an azure-foundry endpoint must refresh credentials and
retry the same provider after a pause, instead of exhausting the fallback ladder in
under a second.

Proven red on base: ``_AUTH_REFRESH_PROVIDER_BY_HOST`` had no ``services.ai.azure.com``
entry, so ``_auth_refresh_provider_for_route("auto", <foundry url>)`` returned ``None``
and the credential-refresh rung never fired; ``_refresh_provider_credentials`` had no
``azure-foundry`` entry and returned False for it.
"""

from __future__ import annotations

FOUNDRY_URL = "https://app-foundry-dev.services.ai.azure.com/openai/v1"


class TestAzureFoundryAuthRecovery:
    def test_foundry_host_maps_to_auth_refresh_provider(self):
        """An azure-foundry host must resolve to a refresh provider, or a 401 from
        Entra-ID auth skips credential recovery entirely."""
        from agent.auxiliary_client import _auth_refresh_provider_for_route

        assert _auth_refresh_provider_for_route("auto", FOUNDRY_URL) == "azure-foundry"

    def test_foundry_is_a_registered_credential_refresher(self):
        """``_refresh_provider_credentials("azure-foundry")`` must attempt the Entra
        chain rather than returning False as an unknown provider."""
        from agent import auxiliary_client

        assert "azure-foundry" in auxiliary_client._CREDENTIAL_REFRESHERS

    def test_refresh_probes_entra_chain(self, monkeypatch):
        """The refresher succeeds when the Entra chain yields a token and fails closed
        when it cannot."""
        from agent import auxiliary_client

        class _OkCred:
            def get_token(self, scope):
                return object()

        monkeypatch.setattr(
            "azure.identity.DefaultAzureCredential", _OkCred, raising=False
        )
        assert auxiliary_client._refresh_azure_foundry_credentials() is True

        class _BadCred:
            def get_token(self, scope):
                raise RuntimeError("no login")

        monkeypatch.setattr(
            "azure.identity.DefaultAzureCredential", _BadCred, raising=False
        )
        assert auxiliary_client._refresh_azure_foundry_credentials() is False

    def test_auth_retry_pauses_ten_seconds_for_foundry(self):
        """The same-provider retry after a credential refresh must wait long enough for
        token-endpoint propagation (10s for azure-foundry), not re-fire instantly."""
        from agent.auxiliary_client import _auth_backoff_delay

        assert _auth_backoff_delay("azure-foundry") == 10.0
        assert _auth_backoff_delay("openai") == 1.0
        assert _auth_backoff_delay(None) == 1.0
        assert _auth_backoff_delay("auto") == 1.0

    def test_retry_ladder_step_resolves_to_same_provider_with_delay(self):
        """A ``retry_same_provider`` step for azure-foundry must map to the retry
        performer carrying the foundry provider id — the performer owns the 10s pause —
        and must not sleep during step resolution."""
        from agent.auxiliary_client import (
            _PreparedAuxRequest,
            _LadderStep,
            _ladder_step_call,
            _auth_backoff_delay,
        )

        req = _PreparedAuxRequest(
            client=object(), final_model="m", kwargs={}, resolved_provider="azure-foundry",
            request_provider="azure-foundry", resolved_model="m", resolved_base_url=FOUNDRY_URL,
            resolved_api_key=None, resolved_api_mode="chat_completions",
            effective_timeout=30.0, effective_extra_body={}, base_info=FOUNDRY_URL,
        )
        step = _LadderStep("retry_same_provider", ("azure-foundry", "m"))
        kind, _args, kw = _ladder_step_call(step, req, {"max_tokens": 8}, {})
        assert kind == "retry"
        assert kw["resolved_provider"] == "azure-foundry"
        # The delay the performer applies to this step.
        assert _auth_backoff_delay(kw.get("resolved_provider")) == 10.0
