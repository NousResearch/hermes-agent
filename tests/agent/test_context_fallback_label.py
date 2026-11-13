"""A context window Hermes only defaulted to is labelled "fallback" wherever it is shown, so an operator can
tell it from a known or detected limit (field report: GPT-6.1 Terra on Bedrock showed "128K (detected)")."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

TERRA_CARD_WINDOW = 1_050_000
RUNTIME_URL = "https://bedrock-runtime.us-east-1.amazonaws.com"


UNKNOWN = "acme.unknown-model-v1"
LOCAL_URL = "http://127.0.0.1:8000/v1"


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    from hermes_agent.agent import bedrock_adapter, context_default_record
    bedrock_adapter.reset_client_cache()
    monkeypatch.setattr(context_default_record, "_RECORDS", {})
    monkeypatch.setattr("hermes_agent.agent.bedrock_adapter.probe_bedrock_context_length", lambda *a, **k: None)
    yield
    bedrock_adapter.reset_client_cache()


def _resolve(model=UNKNOWN, provider="bedrock", base_url=RUNTIME_URL, config_context_length=None) -> int:
    """The real resolver, as the compressor / gateway / /model call it (it records a defaulted answer)."""
    from hermes_agent.agent.model_metadata import get_model_context_length
    return get_model_context_length(model, base_url=base_url, provider=provider,
                                    config_context_length=config_context_length)


def _resolve_and_mark(model=UNKNOWN, provider="bedrock", base_url=RUNTIME_URL, config_context_length=None):
    from hermes_agent.agent.context_pin import context_window_marker
    ctx = _resolve(model, provider, base_url, config_context_length)
    return ctx, context_window_marker(ctx, config_context_length, model=model, provider=provider, base_url=base_url)


@pytest.fixture
def _quiet_custom_endpoint(monkeypatch):
    """A custom endpoint whose live probes report nothing (tests opt in to a /models answer)."""
    for name in ("_resolve_endpoint_context_length", "_probe_local_context_length", "_query_ollama_api_show"):
        monkeypatch.setattr(f"hermes_agent.agent.model_metadata.{name}", lambda *a, **k: None)
    return monkeypatch


class TestFallbackMarker:
    def test_unknown_bedrock_model_window_is_marked_fallback(self):
        assert _resolve_and_mark() == (128_000, "fallback")

    def test_card_sized_and_pinned_windows_are_not_marked_fallback(self):
        assert _resolve_and_mark("us.openai.gpt-6.1-terra") == (TERRA_CARD_WINDOW, "")
        assert _resolve_and_mark("amazon.nova-micro-v1:0") == (128_000, "")  # table row of exactly 128K
        assert _resolve_and_mark(config_context_length=128_000) == (128_000, "pinned")

    def test_provider_confirmed_window_equal_to_the_default_is_not_marked_fallback(self):
        with patch("hermes_agent.agent.model_metadata.get_cached_context_length", return_value=128_000):
            assert _resolve_and_mark() == (128_000, "")

    def test_a_window_no_resolution_defaulted_is_not_marked_fallback(self):
        from hermes_agent.agent.context_pin import context_window_marker
        _resolve()  # the Bedrock route defaulted; another route showing the same number did not
        assert context_window_marker(128_000, None, model=UNKNOWN, provider="openrouter",
                                     base_url="https://openrouter.ai/api/v1") == ""

    def test_generic_default_for_an_uncatalogued_model_is_marked_fallback(self, _quiet_custom_endpoint):
        from hermes_agent.agent.model_metadata import DEFAULT_FALLBACK_CONTEXT
        route = dict(provider="custom", base_url=LOCAL_URL)
        assert _resolve_and_mark("mystery-model", **route) == (DEFAULT_FALLBACK_CONTEXT, "fallback")
        # Same number from the catalog (a listed 256K model) is a known limit.
        assert _resolve_and_mark("gemma-4-31b", **route) == (DEFAULT_FALLBACK_CONTEXT, "")

    def test_a_window_the_endpoint_reported_is_not_marked_fallback(self, _quiet_custom_endpoint):
        """Greptile: an uncatalogued model whose /models reports 256000 is detected, not defaulted."""
        from hermes_agent.agent.model_metadata import DEFAULT_FALLBACK_CONTEXT
        route = dict(provider="custom", base_url=LOCAL_URL)
        assert _resolve_and_mark("mystery-model", **route)[1] == "fallback"  # endpoint down: defaulted
        _quiet_custom_endpoint.setattr("hermes_agent.agent.model_metadata._resolve_endpoint_context_length",
                                       lambda *a, **k: DEFAULT_FALLBACK_CONTEXT)
        assert _resolve_and_mark("mystery-model", **route) == (DEFAULT_FALLBACK_CONTEXT, "")  # reported: detected

    @pytest.mark.parametrize("defaulted_first", [True, False])
    def test_profiles_keep_their_own_label(self, tmp_path, defaulted_first):
        """Greptile: a multiplexed gateway resolves for several profiles; one profile's answer is not another's."""
        from hermes_agent.agent.context_pin import context_window_marker
        from hermes_agent.hermes_constants import reset_hermes_home_override, set_hermes_home_override
        homes = {"defaulted": tmp_path / "a", "confirmed": tmp_path / "b"}
        for home in homes.values():
            home.mkdir()

        def in_home(name, fn):
            token = set_hermes_home_override(homes[name])
            try:
                return fn()
            finally:
                reset_hermes_home_override(token)

        def confirmed():
            with patch("hermes_agent.agent.model_metadata.get_cached_context_length", return_value=128_000):
                return _resolve()
        order = [("defaulted", _resolve), ("confirmed", confirmed)]
        for name, fn in order if defaulted_first else order[::-1]:
            assert in_home(name, fn) == 128_000
        mark = lambda: context_window_marker(128_000, None, model=UNKNOWN, provider="bedrock", base_url=RUNTIME_URL)
        assert in_home("defaulted", mark) == "fallback"
        assert in_home("confirmed", mark) == ""


class TestFallbackSourceIsOwnedByTheCompressor:
    """Greptile on f6e7d756: the label must follow the lookup that produced the window, not the route's latest."""

    def test_a_later_lookup_on_the_same_route_keeps_a_live_compressors_label(self, _quiet_custom_endpoint):
        from hermes_agent.agent.context_compressor import ContextCompressor
        from hermes_agent.agent.context_pin import agent_context_marker
        from hermes_agent.agent.model_metadata import DEFAULT_FALLBACK_CONTEXT
        compressor = ContextCompressor(model="mystery-model", base_url=LOCAL_URL, provider="custom", quiet_mode=True)
        agent = SimpleNamespace(model="mystery-model", provider="custom", base_url=LOCAL_URL, _config_context_length=None,
                                context_compressor=compressor)
        assert compressor.context_length == DEFAULT_FALLBACK_CONTEXT  # endpoint down: this agent runs on the default
        assert agent_context_marker(agent) == "fallback"
        _quiet_custom_endpoint.setattr("hermes_agent.agent.model_metadata._resolve_endpoint_context_length",
                                       lambda *a, **k: DEFAULT_FALLBACK_CONTEXT)
        assert _resolve_and_mark("mystery-model", provider="custom", base_url=LOCAL_URL)[1] == ""  # another session
        assert agent_context_marker(agent) == "fallback"  # still running on its own defaulted window

    def test_a_moa_preset_keeps_its_aggregators_fallback(self, monkeypatch):
        """The preset's window is the aggregator's; when that lookup defaulted, so did the preset's."""
        from hermes_agent.agent import model_metadata
        from hermes_agent.agent.context_pin import context_window_marker

        def _aggregator(model, custom_providers):
            return model_metadata.get_model_context_length(UNKNOWN, base_url=RUNTIME_URL, provider="bedrock")
        monkeypatch.setattr(model_metadata, "_resolve_moa_context_length", _aggregator)
        ctx = model_metadata.get_model_context_length("my-preset", provider="moa")
        assert ctx == 128_000
        assert context_window_marker(ctx, None, model="my-preset", provider="moa") == "fallback"


class TestFallbackIsLabelledOnSurfaces:
    """Every surface that shows the window says when it is the fallback, not a known/detected limit."""

    def test_gateway_session_info_says_fallback_not_detected(self, tmp_path):
        from hermes_agent.gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        (tmp_path / "config.yaml").write_text("model:\n  default: acme.unknown-model-v1\n  provider: bedrock\n")
        runtime = {"provider": "bedrock", "base_url": RUNTIME_URL, "api_key": "aws-sdk"}
        with patch("hermes_agent.gateway.run._hermes_home", tmp_path), \
                patch("hermes_agent.gateway.run._resolve_gateway_model", return_value="acme.unknown-model-v1"), \
                patch("hermes_agent.gateway.run._resolve_runtime_agent_kwargs", return_value=runtime):
            info = runner._format_session_info()
        line = next(x for x in info.splitlines() if "128K" in x)
        assert "(fallback" in line and "(detected)" not in line
        assert "model.context_length" in line

    def _cli_with_bedrock_agent(self, model="acme.unknown-model-v1", context_length=128_000):
        from datetime import datetime, timedelta
        from hermes_agent.cli import HermesCLI
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.model = model
        cli_obj.session_start = datetime.now() - timedelta(minutes=3)
        cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
        cli_obj.agent = SimpleNamespace(
            model=model, provider="bedrock", base_url=RUNTIME_URL, _config_context_length=None,
            session_input_tokens=9_000, session_output_tokens=1_000, session_cache_read_tokens=0,
            session_cache_write_tokens=0, session_prompt_tokens=9_000, session_completion_tokens=1_000,
            session_total_tokens=10_000, session_api_calls=2, get_rate_limit_state=lambda: None,
            context_compressor=SimpleNamespace(last_prompt_tokens=9_000, context_length=context_length,
                                               compression_count=0, _config_context_length=None),
        )
        cli_obj._print_account_limits = lambda: False
        cli_obj._print_nous_credits_block = lambda: False
        cli_obj.verbose = False
        return cli_obj

    def test_cli_usage_marks_the_fallback_window(self, capsys):
        _resolve()
        self._cli_with_bedrock_agent()._show_usage()
        line = next(x for x in capsys.readouterr().out.splitlines() if "128,000" in x)
        assert "(fallback)" in line

    def test_cli_usage_leaves_a_card_sized_window_unmarked(self, capsys):
        self._cli_with_bedrock_agent("us.openai.gpt-6.1-terra", TERRA_CARD_WINDOW)._show_usage()
        assert "(fallback)" not in capsys.readouterr().out

    def test_cli_status_bar_marks_the_fallback_window(self):
        _resolve()
        text = self._cli_with_bedrock_agent()._build_status_bar_text(width=120)
        assert "/128K fallback" in text

    def test_banner_marks_the_fallback_window(self):
        from hermes_agent.hermes_cli.banner import _banner_left_lines
        lines = _banner_left_lines("acme.unknown-model-v1", "/tmp", None, 128_000, "bedrock", accent="#fff",
                                   dim="#888", context_marker="fallback")
        assert any("128K context (fallback)" in str(x) for x in lines)

    @pytest.mark.asyncio
    async def test_gateway_usage_marks_the_fallback_window(self):
        _resolve()
        agent = MagicMock(model="acme.unknown-model-v1", provider="bedrock", base_url=RUNTIME_URL,
                          session_input_tokens=9_000, session_output_tokens=1_000, session_total_tokens=10_000,
                          session_api_calls=2, session_cache_read_tokens=0, session_cache_write_tokens=0)
        agent.get_rate_limit_state.return_value = None
        agent.context_compressor = SimpleNamespace(last_prompt_tokens=9_000, context_length=128_000,
                                                   compression_count=0, _config_context_length=None)
        from hermes_agent.gateway.slash_commands_status import _usage_agent_stats_lines
        line = next(x for x in _usage_agent_stats_lines(agent) if "128,000" in x)
        assert "(fallback)" in line

    def test_cli_model_switch_marks_the_fallback_window(self, monkeypatch):
        import hermes_agent.cli as cli_mod
        from hermes_agent.hermes_cli.model_switch import ModelSwitchResult
        captured: list[str] = []
        monkeypatch.setattr(cli_mod, "_cprint", lambda s, *a, **k: captured.append(str(s)))
        stub = SimpleNamespace(agent=None, base_url="", api_key="", config={}, _pending_model_switch_note="")
        result = ModelSwitchResult(
            success=True, new_model="acme.unknown-model-v1", target_provider="bedrock", provider_changed=True,
            api_key="aws-sdk", base_url=RUNTIME_URL, api_mode="bedrock_converse", warning_message="",
            provider_label="AWS Bedrock", resolved_via_alias=False, capabilities=None, model_info=None,
            is_global=False,
        )
        from hermes_agent.hermes_cli.cli_model_switch_mixin import _print_switch_summary
        _print_switch_summary(stub, result, "us.openai.gpt-6-sol", one_turn=False, strict_context=True)
        line = next(x for x in captured if "128,000" in x)
        assert "(fallback)" in line
