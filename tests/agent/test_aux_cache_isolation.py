"""Regression: auxiliary_client module-level runtime caches must not leak
across tests in a single-process (multi-file) pytest run.

Under the canonical runner (``scripts/run_tests_parallel.py``) each test
FILE gets its own subprocess, so these module globals can't leak across
files. But a plain ``pytest tests/agent/`` (or any single-process run that
touches multiple files) shares one interpreter — and three runtime caches
in ``agent.auxiliary_client`` would otherwise carry state from one test
into the next:

  * ``_aux_unhealthy_until`` / ``_aux_unhealthy_logged_at`` — the
    "recently 402'd" provider cache. A real ``AIAgent`` construction in an
    earlier compressor/agent test probes providers, marks ``nous`` /
    ``openrouter`` unhealthy with a 600s TTL, and that mark then makes
    ``_resolve_auto`` skip the (mocked) provider in a later
    ``test_auxiliary_main_first`` test → it returns ``None`` and the test
    fails. This was the concrete 5-failure cascade that motivated the fix.
  * ``_RUNTIME_MAIN_*`` runtime override (via ``set_runtime_main`` /
    ``clear_runtime_main``).
  * ``_client_cache`` — resolved provider clients keyed by config.

The hermetic autouse fixture in ``tests/conftest.py`` resets all three at
the start of every test. These tests prove that contract: a test that
dirties the caches must not affect the next test.
"""

from __future__ import annotations

import agent.auxiliary_client as aux


class TestAuxUnhealthyCacheIsolation:
    """The unhealthy-provider cache must be empty at the start of each test."""

    def test_a_dirty_the_unhealthy_cache(self):
        # Simulate what a real AIAgent construction does when a provider 402s.
        aux._mark_provider_unhealthy("nous", ttl=600)
        aux._mark_provider_unhealthy("openrouter", ttl=600)
        assert aux._is_provider_unhealthy("nous") is True
        assert aux._is_provider_unhealthy("openrouter") is True

    def test_b_cache_is_clean_for_next_test(self):
        # If the hermetic fixture did NOT reset between tests, the marks from
        # test_a would still be live here (600s TTL) and these would be True.
        assert aux._is_provider_unhealthy("nous") is False
        assert aux._is_provider_unhealthy("openrouter") is False
        assert aux._aux_unhealthy_until == {}
        assert aux._aux_unhealthy_logged_at == {}


class TestRuntimeMainIsolation:
    """The runtime-main override must not leak across tests."""

    def test_a_set_runtime_main(self):
        aux.set_runtime_main(provider="nous", model="anthropic/claude-opus-4.6")
        assert aux._RUNTIME_MAIN_PROVIDER == "nous"
        assert aux._RUNTIME_MAIN_MODEL == "anthropic/claude-opus-4.6"

    def test_b_runtime_main_is_clear_for_next_test(self):
        assert aux._RUNTIME_MAIN_PROVIDER == ""
        assert aux._RUNTIME_MAIN_MODEL == ""


class TestClientCacheIsolation:
    """The resolved-client cache must not leak across tests."""

    def test_a_dirty_the_client_cache(self):
        aux._client_cache[("sentinel", False, "", "", "", "")] = (
            object(),
            "sentinel-model",
            None,
        )
        assert len(aux._client_cache) >= 1

    def test_b_client_cache_is_clean_for_next_test(self):
        assert aux._client_cache == {}
