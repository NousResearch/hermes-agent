"""Tests for NetworkMixin -- network/client methods extracted from AIAgent."""

import pytest


class TestNetworkMixinImport:
    def test_mixin_importable(self):
        from agent.network import NetworkMixin
        assert NetworkMixin is not None

    def test_has_get_transport(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_get_transport')

    def test_has_build_keepalive_http_client(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_build_keepalive_http_client')

    def test_has_force_close_tcp_sockets(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_force_close_tcp_sockets')

    def test_has_apply_client_headers_for_base_url(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_apply_client_headers_for_base_url')

    def test_has_rebuild_anthropic_client(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_rebuild_anthropic_client')

    def test_has_try_refresh_nous_client_credentials(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_try_refresh_nous_client_credentials')

    def test_has_try_refresh_copilot_client_credentials(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_try_refresh_copilot_client_credentials')

    def test_has_try_refresh_codex_client_credentials(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_try_refresh_codex_client_credentials')

    def test_has_credential_pool_may_recover_rate_limit(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_credential_pool_may_recover_rate_limit')

    def test_has_swap_credential(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_swap_credential')

    def test_has_is_azure_openai_url(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_is_azure_openai_url')

    def test_has_is_direct_openai_url(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_is_direct_openai_url')

    def test_has_is_github_copilot_url(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_is_github_copilot_url')

    def test_has_is_openrouter_url(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_is_openrouter_url')

    def test_has_resolved_api_call_stale_timeout_base(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_resolved_api_call_stale_timeout_base')

    def test_has_resolved_api_call_timeout(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_resolved_api_call_timeout')

    def test_has_mask_api_key_for_logs(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_mask_api_key_for_logs')

    def test_has_check_openrouter_cache_status(self):
        from agent.network import NetworkMixin
        assert hasattr(NetworkMixin, '_check_openrouter_cache_status')
