"""A provider-less session ``/model`` override must not post a foreign model to the turn's provider.

``/model <name>`` without ``--provider`` records only the model. The gateway then keeps the default
provider's key and endpoint, so a model that provider does not serve is sent to the wrong API and
fails with a non-retryable 400 (e.g. a non-Codex model posted to the ChatGPT/Codex backend). The
failed runtime is written back into the session row afterwards, which makes every later turn repeat
the same failure until the model is re-picked.

These tests exercise the real ``_apply_session_model_override()`` with the catalog helpers stubbed,
so no credentials or network are involved.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource, build_session_key

CONFIG_MODEL = "gpt-6.1-sol-900k"
DEFAULT_PROVIDER = "openai-codex"
CODEX_RUNTIME = {
    "provider": DEFAULT_PROVIDER,
    "api_key": "test-key",
    "base_url": "https://chatgpt.com/backend-api/codex",
    "api_mode": "codex_responses",
}


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_runner(override: dict):
    """Minimal GatewayRunner whose only live state is the session's /model override."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")}
    )
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._pre_agent_fallback_notice = None
    runner._peek_session_state = lambda _skey: SimpleNamespace(
        conversation=SimpleNamespace(model_override=override)
    )
    return runner


def _apply(override: dict, *, catalog, owns_vendor=False, static_hit=False, model=CONFIG_MODEL):
    runner = _make_runner(override)
    sk = build_session_key(_make_source())
    with patch("hermes_cli.models.cached_provider_model_ids", return_value=list(catalog)), \
            patch("hermes_cli.models_detect.current_provider_owns_vendor", return_value=owns_vendor), \
            patch("hermes_cli.models._model_in_provider_catalog", return_value=static_hit), \
            patch("hermes_cli.models._provider_keys",
                  return_value={DEFAULT_PROVIDER, "openai-api", "openai"}):
        applied_model, runtime = runner._apply_session_model_override(sk, model, dict(CODEX_RUNTIME))
    return runner, applied_model, runtime


class TestProviderLessOverride:
    def test_model_the_provider_does_not_serve_is_not_applied(self):
        """The reported shape: DeepSeek id recorded with no provider on a Codex-only session."""
        runner, model, runtime = _apply({"model": "deepseek-v4.1-flash"}, catalog=["gpt-6-luna"])
        assert model == CONFIG_MODEL
        assert runtime == CODEX_RUNTIME
        assert "deepseek-v4.1-flash" in runner._pre_agent_fallback_notice
        assert "not served by that provider" in runner._pre_agent_fallback_notice

    def test_model_the_provider_serves_is_applied(self):
        _, model, runtime = _apply({"model": "gpt-6-luna"}, catalog=["gpt-6.1-sol-900k", "gpt-6-luna"])
        assert model == "gpt-6-luna"
        assert runtime == CODEX_RUNTIME

    def test_vendor_bare_part_after_slash_is_applied(self):
        _, model, _ = _apply({"model": "openai/gpt-6-luna"}, catalog=["openai/gpt-6-luna"])
        assert model == "openai/gpt-6-luna"

    def test_unknown_but_statically_listed_model_is_applied(self):
        """Aliases and early-access ids missing from the live catalog stay the provider's call."""
        _, model, _ = _apply({"model": "gpt-6.1-sol-900k"}, catalog=["gpt-6-luna"], static_hit=True)
        assert model == "gpt-6.1-sol-900k"

    def test_vendor_owned_id_is_applied(self):
        _, model, _ = _apply({"model": "gpt-6.1-local-alias"}, catalog=["gpt-6-luna"], owns_vendor=True)
        assert model == "gpt-6.1-local-alias"

    def test_empty_catalog_never_blocks_the_pick(self):
        """A cold or failed catalog fetch must not silently drop the user's choice."""
        _, model, _ = _apply({"model": "deepseek-v4.1-flash"}, catalog=[])
        assert model == "deepseek-v4.1-flash"

    def test_override_that_names_its_provider_is_untouched(self):
        override = {"model": "deepseek-v4.1-flash", "provider": "opencode-go"}
        runner, model, runtime = _apply(override, catalog=["gpt-6-luna"])
        assert model == "deepseek-v4.1-flash"
        assert runtime["provider"] == "opencode-go"
        assert runner._pre_agent_fallback_notice is None
