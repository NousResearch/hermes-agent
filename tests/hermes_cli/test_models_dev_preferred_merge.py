"""Tests for the models.dev-preferred merge behavior in provider_model_ids
and list_authenticated_providers.

These guard the contract:

  * For providers in ``_MODELS_DEV_PREFERRED`` (opencode-go, opencode-zen,
    xiaomi, deepseek, smaller inference providers), both the CLI model
    picker path (``provider_model_ids``) and the gateway ``/model`` picker
    path (``list_authenticated_providers``) merge fresh models.dev entries
    on top of the curated static list.
  * OpenRouter and Nous Portal are NEVER merged — they keep their curated
    (OpenRouter) or live-Portal (Nous) semantics.
  * If models.dev is unreachable (offline / CI), the curated list is the
    fallback — no crash, no empty list.

Merging is what lets new models (e.g. ``mimo-v2.5-pro`` on opencode-go)
appear in ``/model`` without a Hermes release.
"""

from unittest.mock import patch


from hermes_cli.models import (
    _MODELS_DEV_PREFERRED,
    _PROVIDER_MODELS,
    _merge_with_models_dev,
    provider_model_ids,
)


class TestMergeHelper:
    def test_merge_empty_mdev_returns_curated(self):
        """When models.dev returns nothing, curated list is preserved verbatim."""
        with patch("agent.models_dev.list_agentic_models", return_value=[]):
            out = _merge_with_models_dev("opencode-go", ["mimo-v2-pro", "kimi-k2.6"])
        assert out == ["mimo-v2-pro", "kimi-k2.6"]


    def test_merge_case_insensitive_dedup(self):
        """Dedup is case-insensitive but preserves the first occurrence's casing."""
        mdev = ["MiniMax-M2.7"]
        curated = ["minimax-m2.7", "minimax-m2.5"]
        with patch("agent.models_dev.list_agentic_models", return_value=mdev):
            out = _merge_with_models_dev("minimax", curated)
        # models.dev casing wins since it came first
        assert out == ["MiniMax-M2.7", "minimax-m2.5"]


class TestProviderModelIdsPreferred:





    def test_k3_live_discovery_is_scoped_to_kimi_coding_endpoint(self):
        """Coding keys discover K3; legacy Moonshot keys must not advertise it."""

        class Response:
            def __init__(self, body: bytes):
                self._body = body

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        def fake_open(req, **_kwargs):
            if req.full_url == "https://api.kimi.com/coding/v1/models":
                return Response(b'{"data":[{"id":"k3"}]}')
            if req.full_url == "https://api.moonshot.ai/v1/models":
                return Response(b'{"data":[{"id":"K3"},{"id":"kimi-k2.6"}]}')
            if req.full_url == "https://example.invalid/v1/models":
                return Response(b'{"data":[{"id":"k3"},{"id":"kimi-k2.6"}]}')
            raise AssertionError(f"unexpected Kimi models URL: {req.full_url}")

        with patch("hermes_cli.urllib_security.open_credentialed_url", side_effect=fake_open):
            with patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={
                    "api_key": "sk-kimi-test",
                    "base_url": "https://api.kimi.com/coding",
                },
            ):
                coding_models = provider_model_ids("kimi-coding")

            with patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={
                    "api_key": "legacy-test",
                    "base_url": "https://api.moonshot.ai/v1",
                },
            ):
                legacy_models = provider_model_ids("kimi-coding")

            with patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={
                    "api_key": "custom-test",
                    "base_url": "https://example.invalid/v1",
                },
            ):
                custom_models = provider_model_ids("kimi-coding")

        # The live bare wire id ``k3`` folds into the curated public slug
        # ``kimi-k3`` (picker alias dedup) — one row, curated slug leads.
        assert coding_models[0] == "kimi-k3"
        assert all(model.lower() != "k3" for model in coding_models)
        assert all(model.lower() != "k3" for model in legacy_models)
        assert all(model.lower() != "k3" for model in custom_models)
        # Legacy / custom endpoints never advertise the k3 family at all
        # via live discovery (their curated floor may still carry kimi-k3).

    def test_kimi_setup_flow_uses_same_coding_plan_catalog(self):
        """The setup wizard must not carry a stale duplicate Kimi model list."""
        from hermes_cli.model_setup_flows import _model_flow_kimi

        captured = {}

        def fake_select(model_list, **_kwargs):
            captured["models"] = model_list
            return None

        with (
            patch("hermes_cli.main_provider_setup._prompt_api_key", return_value=("sk-kimi-test", False)),
            patch("hermes_cli.auth._prompt_model_selection", side_effect=fake_select),
            patch("hermes_cli.config.get_env_value", return_value=""),
            patch("hermes_cli.config.save_env_value"),
        ):
            _model_flow_kimi({}, current_model="")

        assert captured["models"] == _PROVIDER_MODELS["kimi-coding"]
        assert captured["models"][0] == "kimi-k3"


class TestOpenRouterAndNousUnchanged:
    """Per Teknium: openrouter and nous are NEVER merged with models.dev."""


    def test_openrouter_does_not_call_merge(self):
        """openrouter takes its own live path — merge helper must NOT run."""
        with patch(
            "hermes_cli.models._merge_with_models_dev",
            side_effect=AssertionError("merge should not be called for openrouter"),
        ):
            # Even if model_ids() fails for some other reason, we just care
            # that the merge path isn't invoked.
            try:
                provider_model_ids("openrouter")
            except AssertionError:
                raise
            except Exception:
                pass  # model_ids() may fail in the hermetic test env — that's fine.
