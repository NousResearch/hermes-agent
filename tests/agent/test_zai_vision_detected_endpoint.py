"""Regression test: ZAI vision resolver must honor the endpoint auto-detected
at setup time (provider state's detected_endpoint) instead of only
trying the hardcoded generic endpoints.

Before the fix, a Coding Lite / Coding Plan key — which is only valid on
/api/coding/paas/v4 — had its vision calls routed to the generic
/api/paas/v4 endpoint, producing error 1113 ("insufficient balance").  The
main chat model already used the detected endpoint via the credential pool
(agent/credential_pool.py -> _resolve_zai_base_url); this test locks in the
same behaviour for the vision resolver.

A second regression (review follow-up): the vision resolver must reuse the
shared Z.AI credential/base-URL resolver rather than reading auth.json
directly, so a profile that resolves a *global* Z.AI key but has no local
detected_endpoint still picks up the globally cached endpoint via
``_load_provider_state``'s profile → global-root fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch):
    """Temp HERMES_HOME with auth.json + clean credential env vars."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Strip all credential-shaped env vars so each scenario starts hermetic.
    for k in list(os.environ.keys()):
        if k.endswith("_API_KEY") or k.endswith("_TOKEN"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("GLM_BASE_URL", raising=False)

    return str(hermes_home)


def _write_auth(
    home: str,
    api_key: str,
    base_url: str,
    *,
    include_pool: bool = False,
) -> None:
    """Write an auth.json with a detected_endpoint cache for the given key."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    auth = {
        "version": 1,
        "providers": {
            "zai": {
                "detected_endpoint": {
                    "base_url": base_url,
                    "endpoint_id": "coding-global",
                    "model": "glm-5.2",
                    "label": "Global (Coding Plan)",
                    "key_hash": key_hash,
                }
            }
        },
    }
    if include_pool:
        auth["credential_pool"] = {
            "zai": [
                {
                    "id": "zai-test-key",
                    "label": "Z.AI test key",
                    "auth_type": "api_key",
                    "priority": 0,
                    "source": "manual",
                    "access_token": api_key,
                }
            ]
        }
    with open(os.path.join(home, "auth.json"), "w") as fp:
        json.dump(auth, fp)


class TestZaiVisionDetectedEndpoint:
    def test_detected_coding_endpoint_used_for_vision(self, isolated_home, monkeypatch):
        """The detected /api/coding/paas/v4 URL must be the first candidate
        tried for vision — not the generic /api/paas/v4 that 1113s on a
        coding-plan key.
        """
        api_key = "sk-test-coding-plan-key"
        _write_auth(isolated_home, api_key, "https://api.z.ai/api/coding/paas/v4")
        monkeypatch.setenv("GLM_API_KEY", api_key)

        import agent.auxiliary_client as auxiliary_client
        resolve_vision_provider_client = auxiliary_client.resolve_vision_provider_client

        provider, client, _model = resolve_vision_provider_client(provider="zai")
        assert client is not None, (
            "vision client should resolve with the detected endpoint"
        )
        base_url = str(getattr(client, "base_url", ""))
        assert "coding/paas/v4" in base_url, (
            f"vision should use the detected coding endpoint, got {base_url!r}"
        )

    def test_detected_standard_endpoint_used_for_vision(self, isolated_home, monkeypatch):
        """A standard API key must keep its detected generic Z.AI endpoint."""
        api_key = "sk-test-standard-api-key"
        standard_url = "https://api.z.ai/api/paas/v4"
        _write_auth(isolated_home, api_key, standard_url)
        monkeypatch.setenv("GLM_API_KEY", api_key)

        import agent.auxiliary_client as auxiliary_client

        _provider, client, _model = (
            auxiliary_client.resolve_vision_provider_client(provider="zai")
        )
        assert client is not None
        assert str(getattr(client, "base_url", "")).rstrip("/") == standard_url

    def test_glm_base_url_override_used_for_vision(self, isolated_home, monkeypatch):
        """The canonical GLM_BASE_URL override must outrank hardcoded URLs."""
        api_key = "override-zai-key"
        override_url = "https://custom.example/v4"
        monkeypatch.setenv("GLM_API_KEY", api_key)
        monkeypatch.setenv("GLM_BASE_URL", override_url)

        import agent.auxiliary_client as auxiliary_client

        _provider, client, _model = (
            auxiliary_client.resolve_vision_provider_client(provider="zai")
        )

        assert client is not None
        assert str(getattr(client, "api_key", "")) == api_key
        assert str(getattr(client, "base_url", "")).rstrip("/") == override_url

    def test_canonical_resolver_failure_uses_explicit_hardcoded_fallback(
        self, isolated_home, monkeypatch
    ):
        """An explicit key/base pair must remain usable if canonical resolution fails."""
        api_key = "fallback-zai-key"

        from hermes_cli import auth
        import agent.auxiliary_client as auxiliary_client

        def _raise_resolver(*args, **kwargs):
            raise RuntimeError("resolver failed")

        monkeypatch.setattr(
            auth,
            "resolve_api_key_provider_credentials",
            _raise_resolver,
        )

        _provider, client, _model = (
            auxiliary_client.resolve_vision_provider_client(
                provider="zai",
                api_key=api_key,
            )
        )

        assert client is not None
        assert str(getattr(client, "api_key", "")) == api_key
        assert str(getattr(client, "base_url", "")).rstrip("/") == (
            "https://open.bigmodel.cn/api/paas/v4"
        )

    def test_canonical_failure_tries_hardcoded_candidates_in_order(
        self, isolated_home, monkeypatch
    ):
        """Both hardcoded candidates remain available after resolver failure."""
        api_key = "ordered-fallback-key"
        attempts = []
        fallback_client = object()

        from hermes_cli import auth
        import agent.auxiliary_client as auxiliary_client

        def _raise_resolver(*args, **kwargs):
            raise RuntimeError("resolver failed")

        def _record_candidate(
            provider,
            model=None,
            async_mode=False,
            base_url=None,
            api_key=None,
            **kwargs,
        ):
            attempts.append((base_url, api_key))
            if len(attempts) == 1:
                return None, None
            return fallback_client, model

        monkeypatch.setattr(
            auth,
            "resolve_api_key_provider_credentials",
            _raise_resolver,
        )
        monkeypatch.setattr(auxiliary_client, "_get_cached_client", _record_candidate)

        _provider, client, _model = (
            auxiliary_client.resolve_vision_provider_client(
                provider="zai",
                model="glm-4.6v",
                api_key=api_key,
            )
        )

        assert client is fallback_client
        assert attempts == [
            ("https://open.bigmodel.cn/api/paas/v4", api_key),
            ("https://api.z.ai/api/paas/v4", api_key),
        ]

    def test_canonical_failure_preserves_environment_only_key(
        self, isolated_home, monkeypatch
    ):
        """An environment-only key must survive canonical resolver failure."""
        env_key = "environment-only-zai-key"
        monkeypatch.setenv("GLM_API_KEY", env_key)
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        monkeypatch.delenv("Z_AI_API_KEY", raising=False)
        monkeypatch.delenv("GLM_BASE_URL", raising=False)

        from hermes_cli import auth
        import agent.auxiliary_client as auxiliary_client

        def _raise_resolver(*args, **kwargs):
            raise RuntimeError("resolver failed")

        monkeypatch.setattr(
            auth,
            "resolve_api_key_provider_credentials",
            _raise_resolver,
        )

        _provider, client, _model = (
            auxiliary_client.resolve_vision_provider_client(provider="zai")
        )

        assert client is not None
        assert str(getattr(client, "api_key", "")) == env_key
        assert str(getattr(client, "base_url", "")).rstrip("/") == (
            "https://open.bigmodel.cn/api/paas/v4"
        )

    def test_environment_only_key_uses_hardcoded_candidates_in_order(
        self, isolated_home, monkeypatch
    ):
        """Both fallback candidates receive the same environment-only key."""
        env_key = "environment-fallback-order-key"
        attempts = []
        fallback_client = object()
        monkeypatch.setenv("GLM_API_KEY", env_key)
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        monkeypatch.delenv("Z_AI_API_KEY", raising=False)
        monkeypatch.delenv("GLM_BASE_URL", raising=False)

        from hermes_cli import auth
        import agent.auxiliary_client as auxiliary_client

        def _raise_resolver(*args, **kwargs):
            raise RuntimeError("resolver failed")

        def _record_candidate(
            provider,
            model=None,
            async_mode=False,
            base_url=None,
            api_key=None,
            **kwargs,
        ):
            attempts.append((base_url, api_key))
            if len(attempts) == 1:
                return None, None
            return fallback_client, model

        monkeypatch.setattr(
            auth,
            "resolve_api_key_provider_credentials",
            _raise_resolver,
        )
        monkeypatch.setattr(auxiliary_client, "_get_cached_client", _record_candidate)

        _provider, client, _model = (
            auxiliary_client.resolve_vision_provider_client(
                provider="zai",
                model="glm-4.6v",
            )
        )

        assert client is fallback_client
        assert attempts == [
            ("https://open.bigmodel.cn/api/paas/v4", env_key),
            ("https://api.z.ai/api/paas/v4", env_key),
        ]

    def test_credential_pool_key_selects_matching_cached_endpoint(self, isolated_home, monkeypatch):
        """A pool-only key must validate and use the cached coding endpoint."""
        api_key = "pool-only-zai-key"
        _write_auth(
            isolated_home,
            api_key,
            "https://api.z.ai/api/coding/paas/v4",
            include_pool=True,
        )

        import agent.auxiliary_client as auxiliary_client
        resolve_vision_provider_client = auxiliary_client.resolve_vision_provider_client

        provider, client, _model = resolve_vision_provider_client(provider="zai")
        assert client is not None
        assert str(getattr(client, "api_key", "")) == api_key
        assert "coding/paas/v4" in str(getattr(client, "base_url", ""))

    def test_pool_key_and_endpoint_stay_paired_when_env_key_differs(
        self, isolated_home, monkeypatch
    ):
        """The active pool key must drive both cache validation and the request."""
        pool_key = "pool-zai-key"
        _write_auth(
            isolated_home,
            pool_key,
            "https://api.z.ai/api/coding/paas/v4",
            include_pool=True,
        )
        monkeypatch.setenv("GLM_API_KEY", "different-env-key")

        from hermes_cli import auth
        import agent.auxiliary_client as auxiliary_client
        # Loading a pool also seeds environment credentials, so keep that
        # external probe deterministic while asserting the actual invariant:
        # the selected pool key and its cached endpoint stay paired.
        monkeypatch.setattr(auth, "detect_zai_endpoint", lambda *args, **kwargs: None)

        _provider, client, _model = (
            auxiliary_client.resolve_vision_provider_client(provider="zai")
        )

        assert client is not None
        assert str(getattr(client, "api_key", "")) == pool_key
        assert "coding/paas/v4" in str(getattr(client, "base_url", ""))

    def test_explicit_key_uses_matching_global_cache_over_pool_and_env(
        self, tmp_path: Path, monkeypatch
    ):
        """Vision must pair an explicit key with its own profile-fallback cache."""
        global_home = tmp_path / ".hermes"
        profile_home = global_home / "profiles" / "t"
        profile_home.mkdir(parents=True)

        explicit_key = "explicit-zai-key"
        pool_key = "pool-zai-key"
        env_key = "env-zai-key"
        explicit_url = "https://explicit.example/api/coding/paas/v4"
        _write_auth(str(global_home), explicit_key, explicit_url)
        (profile_home / "auth.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "providers": {},
                    "credential_pool": {
                        "zai": [
                            {
                                "id": "zai-profile-pool-key",
                                "label": "Z.AI profile pool key",
                                "auth_type": "api_key",
                                "priority": 0,
                                "source": "manual",
                                "access_token": pool_key,
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("GLM_API_KEY", env_key)
        monkeypatch.delenv("GLM_BASE_URL", raising=False)

        from hermes_cli import auth
        import agent.auxiliary_client as auxiliary_client
        probed_keys = []

        def _detect(api_key, *args, **kwargs):
            probed_keys.append(api_key)
            return None

        monkeypatch.setattr(auth, "detect_zai_endpoint", _detect)

        _provider, client, _model = (
            auxiliary_client.resolve_vision_provider_client(
                provider="zai",
                api_key=explicit_key,
            )
        )

        assert client is not None
        assert str(getattr(client, "api_key", "")) == explicit_key
        assert str(getattr(client, "base_url", "")).rstrip("/") == explicit_url
        assert explicit_key not in probed_keys

    def test_stale_key_hash_falls_back_to_hardcoded(self, isolated_home, monkeypatch):
        """When the cached detected_endpoint was recorded for a *different* key,
        the hash must not match and the cached coding endpoint must NOT be
        used — resolution falls back to the hardcoded generic URLs so a stale
        entry can never poison resolution.
        """
        _write_auth(isolated_home, "sk-old-key",
                    "https://api.z.ai/api/coding/paas/v4")
        monkeypatch.setenv("GLM_API_KEY", "«redacted:sk-…»")

        from hermes_cli import auth
        import agent.auxiliary_client as auxiliary_client
        monkeypatch.setattr(auth, "detect_zai_endpoint", lambda *args, **kwargs: None)
        resolve_vision_provider_client = auxiliary_client.resolve_vision_provider_client

        provider, client, _model = resolve_vision_provider_client(provider="zai")
        assert client is not None
        base_url = str(getattr(client, "base_url", ""))
        assert "coding/paas/v4" not in base_url, (
            "stale key_hash must NOT serve the cached coding endpoint; "
            f"got {base_url!r}"
        )

    def test_global_auth_fallback_serves_cached_endpoint(
        self, tmp_path: Path, monkeypatch
    ):
        """A profile that resolves a global Z.AI key but has no locally cached
        detected_endpoint must still pick up the globally cached endpoint.

        This is the regression flagged in review: the direct auth.json read
        bypassed ``_load_provider_state``'s profile → global-root fallback
        (the same fallback ``read_credential_pool`` uses). A profile could
        therefore resolve a global Z.AI key but miss its globally cached
        detected_endpoint. Reusing the canonical credential/base-URL resolver
        keeps both reads on the global-aware path.
        """
        import agent.auxiliary_client as auxiliary_client
        from hermes_cli import auth

        # Build a profile-mode directory tree:
        #   <root>/.hermes/auth.json                         ← global (has zai)
        #   <root>/.hermes/profiles/t/auth.json              ← profile (empty)
        global_home = tmp_path / ".hermes"
        profile_home = global_home / "profiles" / "t"
        profile_home.mkdir(parents=True)

        api_key = "global-zai-key"
        # Global auth.json carries both the pool key and cached coding endpoint.
        _write_auth(
            str(global_home),
            api_key,
            "https://api.z.ai/api/coding/paas/v4",
            include_pool=True,
        )
        # Profile auth.json is empty — no local detected_endpoint.
        (profile_home / "auth.json").write_text(
            json.dumps({"version": 1, "providers": {}})
        )

        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        for k in list(os.environ.keys()):
            if k.endswith("_API_KEY") or k.endswith("_TOKEN"):
                monkeypatch.delenv(k, raising=False)
        monkeypatch.delenv("GLM_BASE_URL", raising=False)
        monkeypatch.setattr(
            auth,
            "detect_zai_endpoint",
            lambda *args, **kwargs: pytest.fail(
                "globally cached endpoint should skip live detection"
            ),
        )

        resolve_vision_provider_client = auxiliary_client.resolve_vision_provider_client

        _provider, client, _model = resolve_vision_provider_client(provider="zai")
        assert client is not None, (
            "vision client should resolve via the global-auth fallback"
        )
        assert str(getattr(client, "api_key", "")) == api_key
        base_url = str(getattr(client, "base_url", ""))
        assert "coding/paas/v4" in base_url, (
            "vision should use the globally cached coding endpoint, "
            f"got {base_url!r}"
        )
