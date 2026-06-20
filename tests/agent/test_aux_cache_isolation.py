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


class TestAnthropicKeychainBlocked:
    """The macOS Keychain Claude-Code credential read must be blocked by the
    tests/agent/conftest.py autouse fixture so a dev Mac's live OAuth token
    can't leak into credential-resolution assertions."""

    def test_keychain_read_returns_none_by_default(self):
        # The conftest fixture defaults the adapter's platform to non-Darwin,
        # so the real `security find-generic-password` call never fires.
        import agent.anthropic_adapter as aa

        assert aa._read_claude_code_credentials_from_keychain() is None


class TestRealHomeEnvStripped:
    """HERMES_REAL_HOME must be stripped by the hermetic fixture so a
    developer shell's value can't leak into get_real_home() and defeat a
    test's monkeypatch.setenv("HOME", tmp)."""

    def test_hermes_real_home_not_in_env(self):
        import os

        assert os.getenv("HERMES_REAL_HOME") in (None, "")


class TestModelsDevCacheReset:
    """agent.models_dev._models_dev_cache must be empty at test entry so a
    prior test's tiny SAMPLE_REGISTRY can't poison capability lookups
    (e.g. flip vision-capable model detection)."""

    def test_a_poison_models_dev_cache(self):
        import time
        import agent.models_dev as md

        md._models_dev_cache = {"openai": {"models": {}}}
        md._models_dev_cache_time = time.time()
        assert md._models_dev_cache

    def test_b_models_dev_cache_clean_next_test(self):
        import agent.models_dev as md

        assert md._models_dev_cache == {}
        assert md._models_dev_cache_time == 0


class TestSkinSingletonReset:
    """hermes_cli.skin_engine active-skin globals must reset to the lazy-init
    default so a prior test's non-default skin can't change default-skin
    assertions (e.g. the get_cute_tool_message "┊" tool prefix)."""

    def test_a_switch_skin(self):
        from hermes_cli.skin_engine import set_active_skin, get_active_skin_name

        set_active_skin("ares")
        assert get_active_skin_name() == "ares"

    def test_b_skin_back_to_default(self):
        import hermes_cli.skin_engine as se

        assert se._active_skin is None
        assert se._active_skin_name == "default"


class TestBareKeyEnvStripped:
    """Bare *_KEY credential env vars (CLAUDE_API_PROXY_KEY etc., seeded from
    the real ~/.hermes/.env at import) must be stripped so they don't hijack
    resolve_provider('auto') auto-detection in later tests."""

    def test_bare_key_vars_not_in_env(self):
        import os

        leaked = [
            k for k in os.environ
            if k.endswith("_KEY")
            and not k.endswith(("_ACCESS_KEY", "_PRIVATE_KEY", "_ENCRYPT_KEY", "_AES_KEY"))
        ]
        assert leaked == [], f"bare _KEY credential env vars leaked: {leaked}"


