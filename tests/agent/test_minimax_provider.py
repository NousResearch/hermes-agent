"""Tests for MiniMax provider hardening — context lengths, thinking, catalog, beta headers, transport."""

from unittest.mock import patch




class TestMinimaxM3StaleCacheGuard:
    """Pre-catalog builds resolved M3 via the generic 'minimax' catch-all
    (204,800) and persisted it before the 'minimax-m3' (1M) catalog entry
    existed.  The step-1 cache guard must drop that stale value and re-resolve
    to 1M, while leaving correct M2.x entries (204,800) untouched.
    """




    def test_m2_cache_not_clobbered(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import importlib
        import agent.model_metadata as mm
        importlib.reload(mm)
        base = "https://api.minimaxi.com/anthropic"
        # 204,800 is the CORRECT value for M2.x — guard must not touch it.
        for slug in ("MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1"):
            mm.save_context_length(slug, base, 204_800)
            ctx = mm.get_model_context_length(
                slug, base_url=base, api_key="", provider="minimax-cn"
            )
            assert ctx == 204_800, f"{slug} should stay 204800, got {ctx}"



class TestMinimaxThinkingSupport:
    """Verify that older MiniMax models retain manual thinking.

    MiniMax's Anthropic-compat endpoint officially supports the thinking
    parameter (https://platform.minimax.io/docs/api-reference/text-anthropic-api).
    M2.x should retain manual thinking (type=enabled + budget_tokens); the M3.1
    adaptive-thinking contract is tested separately below.
    """

    def test_minimax_m27_gets_manual_thinking(self):
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert "thinking" in kwargs
        assert kwargs["thinking"]["type"] == "enabled"
        assert "budget_tokens" in kwargs["thinking"]
        # MiniMax should NOT get adaptive thinking or output_config
        assert "output_config" not in kwargs






class TestMinimaxM31AdaptiveEffort:
    """MiniMax-M3.1-Flash-Preview speaks the adaptive contract: ``output_config.effort`` in
    low/medium/high/xhigh/max with a default of ``max``, and thinking that cannot be disabled
    (``thinking: {"type": "disabled"}`` -> HTTP 400 "requires adaptive thinking").
    https://platform.minimax.io/docs/guides/text-generation#thinking

    Before this, M3.1 fell through to the manual budget_tokens path, which MiniMax accepts but
    does not map to effort: every Hermes turn ran at the server default (max), whatever
    ``/reasoning`` said.
    """

    M31 = ("MiniMax-M3.1-Flash-Preview", "minimax/MiniMax-M3.1-Flash-Preview", "MiniMax-M3.1")

    def _kwargs(self, model, reasoning_config):
        from agent.anthropic_adapter import build_anthropic_kwargs
        return build_anthropic_kwargs(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config=reasoning_config,
            base_url="https://api.minimax.io/anthropic",
        )

    def test_m31_sends_the_requested_effort(self):
        for model in self.M31:
            for effort in ("low", "medium", "high", "xhigh", "max"):
                kwargs = self._kwargs(model, {"enabled": True, "effort": effort})
                assert kwargs["output_config"] == {"effort": effort}, (model, effort)
                assert kwargs["thinking"]["type"] == "adaptive", model
                assert "budget_tokens" not in kwargs["thinking"], model

    def test_m31_minimal_maps_to_low_not_the_max_default(self):
        kwargs = self._kwargs("MiniMax-M3.1-Flash-Preview", {"enabled": True, "effort": "minimal"})
        assert kwargs["output_config"] == {"effort": "low"}

    def test_m31_thinking_off_is_never_sent_as_a_disable(self):
        """The disable is a 400 on M3.1; omission is the only 'off' it accepts."""
        kwargs = self._kwargs("MiniMax-M3.1-Flash-Preview", {"enabled": False})
        assert "thinking" not in kwargs
        assert "output_config" not in kwargs

    def test_m3_and_m2_keep_manual_thinking(self):
        for model in ("MiniMax-M3", "minimax/MiniMax-M3", "MiniMax-M2.7", "MiniMax-M3-Flash"):
            kwargs = self._kwargs(model, {"enabled": True, "effort": "low"})
            assert kwargs["thinking"]["type"] == "enabled", model
            assert "output_config" not in kwargs, model

    def test_lookalikes_do_not_match(self):
        from agent.anthropic_endpoints import _model_name_is_minimax_adaptive
        for m in ("MiniMax-M3", "MiniMax-M3-Flash", "MiniMax-M31", "minimax-m30",
                  "MiniMax-M3.10", "minimax-m3.11-Flash", "minimax-m3-10", "minimax-m3-1x",
                  "not-minimax-m3.1", "", None):
            assert _model_name_is_minimax_adaptive(m) is False, m
        for m in ("MiniMax-M3.1-Flash-Preview", "minimax-m3.1", "MiniMax-M3-1", "minimax/minimax-m3.1-x",
                  "vendor/MiniMax-M3.1-20260901", "MiniMax-M3-1-Flash-Preview"):
            assert _model_name_is_minimax_adaptive(m) is True, m


class TestMinimaxBetaHeaders:
    """MiniMax Anthropic-compat endpoints reject fine-grained-tool-streaming beta.

    Verify that build_anthropic_client omits the tool-streaming beta for MiniMax
    (both global and China domains) while keeping it for native Anthropic and
    other third-party endpoints.  Covers the fix for #6510 / #6555.
    """

    _TOOL_BETA = "fine-grained-tool-streaming-2025-05-14"
    _THINKING_BETA = "interleaved-thinking-2025-05-14"

    # -- helper ----------------------------------------------------------

    def _build_and_get_betas(self, api_key, base_url=None):
        """Build client, return the anthropic-beta header string."""
        from agent.anthropic_adapter import build_anthropic_client
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(api_key, base_url=base_url)
            kwargs = mock_sdk.Anthropic.call_args[1]
            headers = kwargs.get("default_headers", {})
            return headers.get("anthropic-beta", "")

    # -- MiniMax global --------------------------------------------------

    def test_minimax_global_omits_tool_streaming(self):
        betas = self._build_and_get_betas(
            "mm-key-123", base_url="https://api.minimax.io/anthropic"
        )
        assert self._TOOL_BETA not in betas
        assert self._THINKING_BETA in betas


    # -- MiniMax China ---------------------------------------------------



    # -- Non-MiniMax keeps full betas ------------------------------------




    # -- _common_betas_for_base_url unit tests ---------------------------







class TestMinimaxApiMode:
    """Verify determine_api_mode returns anthropic_messages for MiniMax providers.

    The MiniMax /anthropic endpoint speaks Anthropic Messages wire format,
    not OpenAI chat completions.  The overlay transport must reflect this
    so that code paths calling determine_api_mode() without a base_url
    (e.g. /model switch) get the correct api_mode.
    """

    def test_minimax_returns_anthropic_messages(self):
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode("minimax") == "anthropic_messages"








class TestMinimaxPreserveDots:
    """Verify that MiniMax model names preserve dots through the Anthropic adapter.

    MiniMax model IDs like 'MiniMax-M2.7' must NOT have dots converted to
    hyphens — the endpoint expects the exact name with dots.
    """

    def test_minimax_provider_preserves_dots(self):
        from types import SimpleNamespace
        agent = SimpleNamespace(provider="minimax", base_url="")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True









    def test_normalize_preserves_m25_free_dot(self):
        from agent.anthropic_message_convert import normalize_model_name
        assert normalize_model_name("minimax-m2.5-free", preserve_dots=True) == "minimax-m2.5-free"





class TestMinimaxSwitchModelCredentialGuard:
    """Verify switch_model() does not leak Anthropic credentials to MiniMax.

    The __init__ path correctly guards against this (line 761), but switch_model()
    must mirror that guard. Without it, /model switch to minimax with no explicit
    api_key would fall back to resolve_anthropic_token() and send Anthropic creds
    to the MiniMax endpoint.
    """

    def test_switch_to_minimax_does_not_resolve_anthropic_token(self):
        """switch_model() should NOT call resolve_anthropic_token() for MiniMax."""
        from unittest.mock import patch, MagicMock

        with patch("run_agent.AIAgent.__init__", return_value=None):
            from run_agent import AIAgent
            agent = AIAgent.__new__(AIAgent)
            agent.provider = "anthropic"
            agent.model = "claude-sonnet-4"
            agent.api_key = "sk-ant-fake"
            agent.base_url = "https://api.anthropic.com"
            agent.api_mode = "anthropic_messages"
            agent._anthropic_base_url = "https://api.anthropic.com"
            agent._anthropic_api_key = "sk-ant-fake"
            agent._is_anthropic_oauth = False
            agent._client_kwargs = {}
            agent.client = None
            agent._anthropic_client = MagicMock()
            agent._fallback_chain = []

        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="sk-ant-leaked") as mock_resolve, \
             patch("agent.anthropic_credentials._is_oauth_token", return_value=False):

            agent.switch_model(
                new_model="MiniMax-M2.7",
                new_provider="minimax",
                api_mode="anthropic_messages",
                api_key="mm-key-123",
                base_url="https://api.minimax.io/anthropic",
            )
            # resolve_anthropic_token should NOT be called for non-Anthropic providers
            mock_resolve.assert_not_called()
            # The key passed to build_anthropic_client should be the MiniMax key
            build_args = mock_build.call_args
            assert build_args[0][0] == "mm-key-123"
