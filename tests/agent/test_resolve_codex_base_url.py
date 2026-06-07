"""Tests for resolve_codex_base_url() — centralized Codex base URL resolution.

Ensures HERMES_CODEX_BASE_URL env var takes priority over pool entry
base_url and DEFAULT_CODEX_BASE_URL across all code paths.

Regression test for: https://github.com/NousResearch/hermes-agent/issues/5875
"""

import os
from unittest import mock

from hermes_cli.auth import DEFAULT_CODEX_BASE_URL, resolve_codex_base_url


class TestResolveCodexBaseUrl:
    """resolve_codex_base_url(pool_base_url=None) priority chain."""

    def test_default_when_no_override(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert resolve_codex_base_url() == DEFAULT_CODEX_BASE_URL

    def test_env_var_takes_priority_over_default(self):
        with mock.patch.dict(os.environ, {"HERMES_CODEX_BASE_URL": "http://localhost:8787/v1"}):
            assert resolve_codex_base_url() == "http://localhost:8787/v1"

    def test_env_var_takes_priority_over_pool(self):
        with mock.patch.dict(os.environ, {"HERMES_CODEX_BASE_URL": "http://localhost:8787/v1"}):
            result = resolve_codex_base_url(pool_base_url="https://chatgpt.com/backend-api/codex")
            assert result == "http://localhost:8787/v1"

    def test_pool_base_url_used_when_no_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            result = resolve_codex_base_url(pool_base_url="https://custom-proxy.example.com/v1")
            assert result == "https://custom-proxy.example.com/v1"

    def test_env_var_stripped_and_rstripped(self):
        with mock.patch.dict(os.environ, {"HERMES_CODEX_BASE_URL": "  http://localhost:8787/v1/  "}):
            assert resolve_codex_base_url() == "http://localhost:8787/v1"

    def test_empty_env_var_falls_through(self):
        with mock.patch.dict(os.environ, {"HERMES_CODEX_BASE_URL": "   "}):
            assert resolve_codex_base_url() == DEFAULT_CODEX_BASE_URL

    def test_empty_pool_base_url_falls_through(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert resolve_codex_base_url(pool_base_url="") == DEFAULT_CODEX_BASE_URL
            assert resolve_codex_base_url(pool_base_url=None) == DEFAULT_CODEX_BASE_URL
