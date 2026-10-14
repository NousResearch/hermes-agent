"""Cross-module wiring contracts for the meta-oauth subscription provider.

Each test pins one integration boundary that the core provider touches
outside hermes_cli/auth_meta.py, mirroring the xai-oauth precedents.
"""

from agent.credential_pool import PooledCredential


def _entry(provider: str, source: str) -> PooledCredential:
    return PooledCredential(
        provider=provider, id="abc123", label="probe", auth_type="oauth", priority=0,
        access_token="SECRET_KEY", refresh_token="SECRET_IDT", source=source)


class TestCredentialPersistence:
    def test_device_code_singleton_row_keeps_secrets_on_disk(self):
        """Singleton-seeded rows (source ``device_code``) are Hermes-owned:
        their tokens must survive pool serialization, like openai-codex and
        xai-oauth. Before this entry existed in _PERSISTABLE_PROVIDER_SOURCES
        the row was sanitized to a fingerprint and the pooled credential was
        dead on the next process start."""
        payload = _entry("meta-oauth", "device_code").to_dict()
        assert payload.get("access_token") == "SECRET_KEY"
        assert payload.get("refresh_token") == "SECRET_IDT"

    def test_borrowed_source_still_fails_closed(self):
        """The allowlist must not open the gate: a non-owned source for the
        same provider is still stripped to a fingerprint."""
        payload = _entry("meta-oauth", "external-file").to_dict()
        assert payload.get("access_token") is None
        assert payload.get("secret_fingerprint")


class TestCredentialRemoval:
    def test_removal_step_clears_auth_store_singleton(self):
        """Credential removal (CLI + dashboard) must find a step that clears
        providers.meta-oauth, or removal leaves the singleton behind and the
        pool re-seeds the deleted credential on next load."""
        from agent.credential_sources import find_removal_step

        step = find_removal_step("meta-oauth", "device_code")
        assert step is not None
        assert step.provider == "meta-oauth"
        assert "providers.meta-oauth" in step.description


class TestAuxiliaryHostInference:
    def test_meta_host_infers_subscription_provider(self):
        """Aux calls inheriting the main route arrive as ``auto``; without a
        host row the pool/refresh rungs cannot infer the provider (the
        api.x.ai -> xai-oauth precedent, #84845)."""
        from agent.auxiliary_client import (
            _AUTH_REFRESH_PROVIDER_BY_HOST, _POOL_PROVIDER_BY_HOST, _provider_for_host)

        assert _provider_for_host("https://api.meta.ai/v1", _POOL_PROVIDER_BY_HOST) == "meta-oauth"
        assert _provider_for_host("https://api.meta.ai/v1", _AUTH_REFRESH_PROVIDER_BY_HOST) == "meta-oauth"
