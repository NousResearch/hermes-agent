"""Tests for CompressionMixin -- context compression methods extracted from AIAgent."""

import pytest


class TestCompressionMixinImport:
    def test_mixin_importable(self):
        from agent.compression import CompressionMixin
        assert CompressionMixin is not None

    def test_has_check_feasibility(self):
        from agent.compression import CompressionMixin
        assert hasattr(CompressionMixin, '_check_compression_model_feasibility')

    def test_has_replay_warning(self):
        from agent.compression import CompressionMixin
        assert hasattr(CompressionMixin, '_replay_compression_warning')

    def test_has_compress_context(self):
        from agent.compression import CompressionMixin
        assert hasattr(CompressionMixin, '_compress_context')
