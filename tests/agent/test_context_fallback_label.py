"""A context window Hermes only defaulted to is labelled "fallback" wherever it is shown, so an operator can
tell it from a known or detected limit (field report: GPT-6.1 Terra on Bedrock showed "128K (detected)")."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

TERRA_CARD_WINDOW = 1_050_000
RUNTIME_URL = "https://bedrock-runtime.us-east-1.amazonaws.com"


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    from agent import bedrock_adapter, context_pin
    bedrock_adapter.reset_client_cache()
    monkeypatch.setattr(context_pin, "_FALLBACK_MEMO", {}, raising=False)
    yield
    bedrock_adapter.reset_client_cache()


def _bedrock_fallback(model="acme.unknown-model-v1", **kw):
    from agent.context_pin import context_window_marker
    from agent.bedrock_adapter import BEDROCK_DEFAULT_CONTEXT_LENGTH
    kw.setdefault("provider", "bedrock")
    kw.setdefault("base_url", RUNTIME_URL)
    return context_window_marker(kw.pop("context_length", BEDROCK_DEFAULT_CONTEXT_LENGTH),
                                 kw.pop("config_context_length", None), model=model, **kw)


class TestFallbackMarker:
    def test_unknown_bedrock_model_window_is_marked_fallback(self):
        assert _bedrock_fallback() == "fallback"

    def test_card_sized_and_pinned_windows_are_not_marked_fallback(self):
        assert _bedrock_fallback("us.openai.gpt-6.1-terra", context_length=TERRA_CARD_WINDOW) == ""
        assert _bedrock_fallback("amazon.nova-micro-v1:0") == ""  # table row of exactly 128K
        assert _bedrock_fallback(config_context_length=128_000) == "pinned"

    def test_provider_confirmed_window_equal_to_the_default_is_not_marked_fallback(self):
        with patch("agent.model_metadata.get_cached_context_length", return_value=128_000):
            assert _bedrock_fallback() == ""

    def test_non_bedrock_route_with_the_bedrock_default_is_not_marked_fallback(self):
        assert _bedrock_fallback(provider="openrouter", base_url="https://openrouter.ai/api/v1") == ""

    def test_generic_default_for_an_uncatalogued_model_is_marked_fallback(self):
        from agent.context_pin import context_window_marker
        from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT
        route = dict(provider="custom", base_url="http://127.0.0.1:8000/v1")
        assert context_window_marker(DEFAULT_FALLBACK_CONTEXT, None, model="mystery-model", **route) == "fallback"
        # Same number from the catalog (a listed 256K model) is a known limit.
        assert context_window_marker(DEFAULT_FALLBACK_CONTEXT, None, model="gemma-4-31b", **route) == ""


class TestFallbackIsLabelledOnSurfaces:
    """Every surface that shows the window says when it is the fallback, not a known/detected limit."""

    def test_gateway_session_info_says_fallback_not_detected(self, tmp_path):
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        (tmp_path / "config.yaml").write_text("model:\n  default: acme.unknown-model-v1\n  provider: bedrock\n")
        runtime = {"provider": "bedrock", "base_url": RUNTIME_URL, "api_key": "aws-sdk"}
        with patch("gateway.run._hermes_home", tmp_path), \
                patch("gateway.run._resolve_gateway_model", return_value="acme.unknown-model-v1"), \
                patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime), \
                patch("agent.bedrock_adapter.probe_bedrock_context_length", return_value=None):
            info = runner._format_session_info()
        line = next(x for x in info.splitlines() if "128K" in x)
        assert "(fallback" in line and "(detected)" not in line
        assert "model.context_length" in line

    def _cli_with_bedrock_agent(self, model="acme.unknown-model-v1", context_length=128_000):
        from datetime import datetime, timedelta
        from cli import HermesCLI
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
        self._cli_with_bedrock_agent()._show_usage()
        line = next(x for x in capsys.readouterr().out.splitlines() if "128,000" in x)
        assert "(fallback)" in line

    def test_cli_usage_leaves_a_card_sized_window_unmarked(self, capsys):
        self._cli_with_bedrock_agent("us.openai.gpt-6.1-terra", TERRA_CARD_WINDOW)._show_usage()
        assert "(fallback)" not in capsys.readouterr().out

    def test_cli_status_bar_marks_the_fallback_window(self):
        text = self._cli_with_bedrock_agent()._build_status_bar_text(width=120)
        assert "/128K fallback" in text

    def test_banner_marks_the_fallback_window(self):
        from hermes_cli.banner import _banner_left_lines
        lines = _banner_left_lines("acme.unknown-model-v1", "/tmp", None, 128_000, "bedrock", accent="#fff",
                                   dim="#888", context_marker="fallback")
        assert any("128K context (fallback)" in str(x) for x in lines)

    @pytest.mark.asyncio
    async def test_gateway_usage_marks_the_fallback_window(self):
        agent = MagicMock(model="acme.unknown-model-v1", provider="bedrock", base_url=RUNTIME_URL,
                          session_input_tokens=9_000, session_output_tokens=1_000, session_total_tokens=10_000,
                          session_api_calls=2, session_cache_read_tokens=0, session_cache_write_tokens=0)
        agent.get_rate_limit_state.return_value = None
        agent.context_compressor = SimpleNamespace(last_prompt_tokens=9_000, context_length=128_000,
                                                   compression_count=0, _config_context_length=None)
        from gateway.slash_commands_status import _usage_agent_stats_lines
        line = next(x for x in _usage_agent_stats_lines(agent) if "128,000" in x)
        assert "(fallback)" in line

    def test_cli_model_switch_marks_the_fallback_window(self, monkeypatch):
        import cli as cli_mod
        from hermes_cli.model_switch import ModelSwitchResult
        captured: list[str] = []
        monkeypatch.setattr(cli_mod, "_cprint", lambda s, *a, **k: captured.append(str(s)))
        stub = SimpleNamespace(agent=None, base_url="", api_key="", config={}, _pending_model_switch_note="")
        result = ModelSwitchResult(
            success=True, new_model="acme.unknown-model-v1", target_provider="bedrock", provider_changed=True,
            api_key="aws-sdk", base_url=RUNTIME_URL, api_mode="bedrock_converse", warning_message="",
            provider_label="AWS Bedrock", resolved_via_alias=False, capabilities=None, model_info=None,
            is_global=False,
        )
        from hermes_cli.cli_model_switch_mixin import _print_switch_summary
        with patch("agent.bedrock_adapter.probe_bedrock_context_length", return_value=None):
            _print_switch_summary(stub, result, "us.openai.gpt-6-sol", one_turn=False, strict_context=True)
        line = next(x for x in captured if "128,000" in x)
        assert "(fallback)" in line
