"""Tests for FallbackMixin -- provider fallback methods extracted from AIAgent."""

import pytest


class TestFallbackMixinImport:
    def test_mixin_importable(self):
        from agent.fallback import FallbackMixin
        assert FallbackMixin is not None

    def test_has_try_activate_fallback(self):
        from agent.fallback import FallbackMixin
        assert hasattr(FallbackMixin, '_try_activate_fallback')

    def test_has_restore_primary_runtime(self):
        from agent.fallback import FallbackMixin
        assert hasattr(FallbackMixin, '_restore_primary_runtime')

    def test_has_try_recover_primary_transport(self):
        from agent.fallback import FallbackMixin
        assert hasattr(FallbackMixin, '_try_recover_primary_transport')

    def test_has_recover_with_credential_pool(self):
        from agent.fallback import FallbackMixin
        assert hasattr(FallbackMixin, '_recover_with_credential_pool')

    def test_has_try_refresh_anthropic_credentials(self):
        from agent.fallback import FallbackMixin
        assert hasattr(FallbackMixin, '_try_refresh_anthropic_client_credentials')
