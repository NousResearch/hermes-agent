"""Unit tests for the Lanseq provider profile.

Covers the five contract areas that matter for a profile-based provider:
registration/discovery, provider resolution, env wiring, mocked model
discovery, and reasoning-effort kwargs (build_api_kwargs_extras) through the
real transport. All tests load Lanseq through real discovery and assert
behavior, never source text.
"""

from __future__ import annotations

import io
import json

import pytest


# --- shared helpers --------------------------------------------------------

def _load_lanseq():
    """Resolve the registered Lanseq profile via real discovery.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    profile in the global provider registry (same approach the DeepSeek test
    uses).
    """
    import model_tools  # noqa: F401  (triggers provider discovery)
    import providers

    profile = providers.get_provider_profile("lanseq")
    assert profile is not None, "lanseq provider profile must be registered"
    return profile


# --- 1. Registration / discovery -------------------------------------------

class TestLanseqDiscovery:
    def test_profile_registered_via_discovery(self):
        _load_lanseq()  # raises the assertion if not discovered

    def test_profile_present_in_list_providers(self):
        import model_tools  # noqa: F401
        import providers

        names = {p.name for p in providers.list_providers()}
        assert "lanseq" in names, f"lanseq missing from list_providers(): {sorted(names)}"


# --- 2. Provider resolution ------------------------------------------------

class TestLanseqResolution:
    def test_resolve_provider_resolves_lanseq(self):
        _load_lanseq()
        from hermes_cli import auth

        assert auth.resolve_provider("lanseq") == "lanseq"


# --- 3. Env wiring (PROVIDER_REGISTRY) -------------------------------------

class TestLanseqEnvWiring:
    def test_api_key_and_base_url_env_vars_split(self):
        _load_lanseq()
        from hermes_cli import auth

        pconfig = auth.PROVIDER_REGISTRY.get("lanseq")
        assert pconfig is not None, "lanseq must be in PROVIDER_REGISTRY"

        # Non-URL env var carries the secret; the *_BASE_URL var overrides base URL.
        assert "LANSEQ_API_KEY" in pconfig.api_key_env_vars
        assert "LANSEQ_BASE_URL" not in pconfig.api_key_env_vars
        assert pconfig.base_url_env_var == "LANSEQ_BASE_URL"


# --- 4. Mocked model discovery ---------------------------------------------

class TestLanseqModelDiscovery:
    def test_fetch_models_parses_openai_compat_catalog(self, monkeypatch):
        profile = _load_lanseq()

        payload = {
            "data": [
                {"id": "qwen3.8-27b-int4"},
                {"id": "qwen3.8-72b"},
                {"no_id": True},  # must be skipped
            ]
        }
        body = json.dumps(payload).encode()

        class _Resp:
            def read(self):
                return body

        class _Ctx:
            def __enter__(self):
                return _Resp()

            def __exit__(self, *a):
                return False

        captured = {}

        def fake_open_credentialed_url(req, timeout=None):
            captured["url"] = req.full_url
            captured["auth"] = req.get_header("Authorization")
            return _Ctx()

        import hermes_cli.urllib_security as sec
        monkeypatch.setattr(sec, "open_credentialed_url", fake_open_credentialed_url)

        models = profile.fetch_models(api_key="test-key-123")
        assert models == ["qwen3.8-27b-int4", "qwen3.8-72b"]

        # Endpoint derived from the profile base_url, Bearer auth forwarded.
        assert captured["url"] == "https://api.lanseq.cloud/v1/models"
        assert captured["auth"] == "Bearer test-key-123"

    def test_fetch_models_returns_none_on_error(self, monkeypatch):
        profile = _load_lanseq()

        import hermes_cli.urllib_security as sec

        def boom(req, timeout=None):
            raise OSError("network down")

        monkeypatch.setattr(sec, "open_credentialed_url", boom)
        assert profile.fetch_models(api_key="k") is None


# --- 5. Reasoning-effort kwargs --------------------------------------------

class TestLanseqReasoningEffort:
    def test_no_reasoning_config_emits_nothing(self):
        profile = _load_lanseq()
        extra_body, top = profile.build_api_kwargs_extras(reasoning_config=None)
        assert extra_body == {}
        assert top == {}

    def test_high_effort_clamped_to_medium(self):
        profile = _load_lanseq()
        _, top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}
        )
        assert top == {"reasoning_effort": "medium"}

    @pytest.mark.parametrize("effort,expected", [
        ("low", "low"),
        ("medium", "medium"),
        ("xhigh", "xhigh"),
        ("max", "xhigh"),
        ("ultra", "xhigh"),
        ("minimal", "none"),
    ])
    def test_effort_map(self, effort, expected):
        profile = _load_lanseq()
        _, top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )
        assert top == {"reasoning_effort": expected}

    def test_disabled_forces_none(self):
        profile = _load_lanseq()
        _, top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}
        )
        assert top == {"reasoning_effort": "none"}

    def test_unknown_effort_omits(self):
        profile = _load_lanseq()
        extra_body, top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "garbage"}
        )
        assert extra_body == {}
        assert top == {}

    def test_default_max_tokens(self):
        profile = _load_lanseq()
        assert profile.get_max_tokens("qwen3.8-27b-int4") == 8192

    def test_full_kwargs_through_transport(self):
        profile = _load_lanseq()
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="qwen3.8-27b-int4",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=profile,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://api.lanseq.cloud/v1",
            provider_name="lanseq",
        )
        assert kwargs["model"] == "qwen3.8-27b-int4"
        # "high" clamps to "medium" on the wire.
        assert kwargs.get("reasoning_effort") == "medium"
