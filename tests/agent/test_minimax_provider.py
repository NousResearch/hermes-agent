"""Tests for MiniMax provider hardening — context lengths, thinking, catalog, beta headers, transport."""

from unittest.mock import patch


class TestMinimaxContextLengths:
    """Verify context length entries match official docs.

    M2.x series is 204,800; M3 is 1M (max output 512K).
    Source: https://platform.minimax.io/docs/api-reference/text-anthropic-api
    """

    def test_minimax_prefix_has_correct_context(self):
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
        assert DEFAULT_CONTEXT_LENGTHS["minimax"] == 204_800

    def test_minimax_models_resolve_via_prefix(self):
        from agent.model_metadata import get_model_context_length
        # M2.x models resolve to 204,800 via the "minimax" catch-all
        for model in ("MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"):
            ctx = get_model_context_length(model, "")
            assert ctx == 204_800, f"{model} expected 204800, got {ctx}"

    def test_minimax_m3_resolves_to_1m(self):
        from agent.model_metadata import get_model_context_length
        # M3 must beat the generic "minimax" catch-all (204,800) and resolve to
        # a 1M-class context. The exact value depends on the source: our
        # hardcoded catalog says 1,000,000; the OpenRouter catalog reports
        # 1,048,576 (1024²). Either is correct — assert "≥ 1M, not 204,800".
        for model in ("MiniMax-M3", "minimax/minimax-m3", "minimax-m3"):
            ctx = get_model_context_length(model, "")
            assert ctx >= 1_000_000, f"{model} expected 1M-class, got {ctx}"


class TestMinimaxM3StaleCacheGuard:
    """Pre-catalog builds resolved M3 via the generic 'minimax' catch-all
    (204,800) and persisted it before the 'minimax-m3' (1M) catalog entry
    existed.  The step-1 cache guard must drop that stale value and re-resolve
    to 1M, while leaving correct M2.x entries (204,800) untouched.
    """

    def test_suggests_minimax_m3(self):
        from agent.model_metadata import _model_name_suggests_minimax_m3
        assert _model_name_suggests_minimax_m3("MiniMax-M3")
        assert _model_name_suggests_minimax_m3("minimax/minimax-m3")
        assert not _model_name_suggests_minimax_m3("MiniMax-M2.7")
        assert not _model_name_suggests_minimax_m3("MiniMax-M2.5")

    def test_stale_m3_cache_dropped_and_reresolves(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import importlib
        import agent.model_metadata as mm
        importlib.reload(mm)
        base = "https://api.minimaxi.com/anthropic"
        mm.save_context_length("MiniMax-M3", base, 204_800)
        ctx = mm.get_model_context_length(
            "MiniMax-M3", base_url=base, api_key="", provider="minimax-cn"
        )
        # Invariant: the stale 204,800 catch-all value must be DROPPED and
        # re-resolved to M3's real, larger context. The exact value depends on
        # the resolution source (hardcoded catalog = 1,000,000; the models.dev
        # registry currently reports 512,000) — both are large-context values
        # well above the generic "minimax" catch-all. Assert the contract
        # ("> 204,800, stale value gone"), not a brittle literal.
        assert ctx > 204_800, f"stale M3 cache not dropped/re-resolved, got {ctx}"

    def test_correct_m3_cache_preserved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import importlib
        import agent.model_metadata as mm
        importlib.reload(mm)
        base = "https://api.minimaxi.com/anthropic"
        mm.save_context_length("MiniMax-M3", base, 1_000_000)
        ctx = mm.get_model_context_length(
            "MiniMax-M3", base_url=base, api_key="", provider="minimax-cn"
        )
        assert ctx == 1_000_000

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
    """Verify that MiniMax gets manual thinking (not adaptive).

    MiniMax's Anthropic-compat endpoint officially supports the thinking
    parameter (https://platform.minimax.io/docs/api-reference/text-anthropic-api).
    It should get manual thinking (type=enabled + budget_tokens), NOT adaptive
    thinking (which is Claude 4.6-only).
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

    def test_minimax_m25_gets_manual_thinking(self):
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="MiniMax-M2.5",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert "thinking" in kwargs
        assert kwargs["thinking"]["type"] == "enabled"

    def test_thinking_still_works_for_claude(self):
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert "thinking" in kwargs


class TestMinimaxAuxModel:
    """Verify auxiliary model is the current frontier standard (not highspeed).

    As of M3's release (2026-06-01) the minimax / minimax-cn provider
    profiles advertise ``MiniMax-M3`` as their ``default_aux_model`` (the
    same model users see in ``_PROVIDER_MODELS["minimax"]`` and in the
    user-facing ``model.default`` for a Token-Plan install).  The OAuth
    / Coding Plan path sticks with M2.7 because M3 is not on that
    tier — see ``test_minimax_profile.py`` for the per-provider split.

    The historical concern this class guards is the #4082 / #6082
    regression: the highspeed variant costs 2x with no model-quality
    benefit, so we still assert that no aux choice contains the substring
    ``"highspeed"``.
    """

    def test_minimax_aux_is_standard(self):
        # Import model_tools to trigger plugin discovery so the
        # ProviderProfile objects are registered in the providers
        # registry before _get_aux_model_for_provider() is called.
        # Without this, profile-based resolution can be order-dependent
        # or fail outright in isolation (the minimax-* entries are
        # no longer in _API_KEY_PROVIDER_AUX_MODELS_FALLBACK after the
        # minimax-M3 default-aux-model cleanup, so the profile is
        # the only path to a non-empty aux value).
        import model_tools  # noqa: F401
        from agent.auxiliary_client import _get_aux_model_for_provider
        assert _get_aux_model_for_provider("minimax") == "MiniMax-M3"
        assert _get_aux_model_for_provider("minimax-cn") == "MiniMax-M3"

    def test_minimax_aux_not_highspeed(self):
        import model_tools  # noqa: F401
        from agent.auxiliary_client import _get_aux_model_for_provider
        assert "highspeed" not in _get_aux_model_for_provider("minimax")
        assert "highspeed" not in _get_aux_model_for_provider("minimax-cn")


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

    def test_minimax_global_trailing_slash(self):
        betas = self._build_and_get_betas(
            "mm-key-123", base_url="https://api.minimax.io/anthropic/"
        )
        assert self._TOOL_BETA not in betas

    # -- MiniMax China ---------------------------------------------------

    def test_minimax_cn_omits_tool_streaming(self):
        betas = self._build_and_get_betas(
            "mm-cn-key-456", base_url="https://api.minimaxi.com/anthropic"
        )
        assert self._TOOL_BETA not in betas
        assert self._THINKING_BETA in betas

    def test_minimax_cn_trailing_slash(self):
        betas = self._build_and_get_betas(
            "mm-cn-key-456", base_url="https://api.minimaxi.com/anthropic/"
        )
        assert self._TOOL_BETA not in betas

    # -- Non-MiniMax keeps full betas ------------------------------------

    def test_native_anthropic_keeps_tool_streaming(self):
        betas = self._build_and_get_betas("sk-ant-api03-real-key-here")
        assert self._TOOL_BETA in betas
        assert self._THINKING_BETA in betas

    def test_third_party_proxy_keeps_tool_streaming(self):
        betas = self._build_and_get_betas(
            "custom-key", base_url="https://my-proxy.example.com/anthropic"
        )
        assert self._TOOL_BETA in betas

    def test_custom_base_url_keeps_tool_streaming(self):
        betas = self._build_and_get_betas(
            "custom-key", base_url="https://custom.api.com"
        )
        assert self._TOOL_BETA in betas

    # -- _common_betas_for_base_url unit tests ---------------------------

    def test_common_betas_none_url(self):
        from agent.anthropic_adapter import _common_betas_for_base_url, _COMMON_BETAS
        assert _common_betas_for_base_url(None) == _COMMON_BETAS

    def test_common_betas_empty_url(self):
        from agent.anthropic_adapter import _common_betas_for_base_url, _COMMON_BETAS
        assert _common_betas_for_base_url("") == _COMMON_BETAS

    def test_common_betas_minimax_url(self):
        from agent.anthropic_adapter import _common_betas_for_base_url, _TOOL_STREAMING_BETA
        betas = _common_betas_for_base_url("https://api.minimax.io/anthropic")
        assert _TOOL_STREAMING_BETA not in betas
        assert len(betas) > 0  # still has other betas

    def test_common_betas_minimax_cn_url(self):
        from agent.anthropic_adapter import _common_betas_for_base_url, _TOOL_STREAMING_BETA
        betas = _common_betas_for_base_url("https://api.minimaxi.com/anthropic")
        assert _TOOL_STREAMING_BETA not in betas

    def test_common_betas_regular_url(self):
        from agent.anthropic_adapter import _common_betas_for_base_url, _COMMON_BETAS
        assert _common_betas_for_base_url("https://api.anthropic.com") == _COMMON_BETAS


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

    def test_minimax_cn_returns_anthropic_messages(self):
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode("minimax-cn") == "anthropic_messages"

    def test_minimax_with_url_also_works(self):
        from hermes_cli.providers import determine_api_mode
        # Even with explicit base_url, provider lookup takes priority
        assert determine_api_mode("minimax", "https://api.minimax.io/anthropic") == "anthropic_messages"

    def test_anthropic_still_returns_anthropic_messages(self):
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode("anthropic") == "anthropic_messages"

    def test_openai_returns_chat_completions(self):
        from hermes_cli.providers import determine_api_mode
        # Sanity check: standard providers are unaffected
        result = determine_api_mode("deepseek")
        assert result == "chat_completions"


class TestMinimaxMaxOutput:
    """Verify _get_anthropic_max_output returns correct limits for MiniMax models.

    MiniMax max output is 131,072 tokens (source: OpenClaw model definitions,
    cross-referenced with MiniMax API behavior).
    """

    def test_minimax_m27_output_limit(self):
        from agent.anthropic_adapter import _get_anthropic_max_output
        assert _get_anthropic_max_output("MiniMax-M2.7") == 131_072

    def test_minimax_m25_output_limit(self):
        from agent.anthropic_adapter import _get_anthropic_max_output
        assert _get_anthropic_max_output("MiniMax-M2.5") == 131_072

    def test_minimax_m2_output_limit(self):
        from agent.anthropic_adapter import _get_anthropic_max_output
        assert _get_anthropic_max_output("MiniMax-M2") == 131_072

    def test_claude_output_unaffected(self):
        from agent.anthropic_adapter import _get_anthropic_max_output
        # Sanity: Claude limits are not broken by the MiniMax entry
        assert _get_anthropic_max_output("claude-sonnet-4-6") == 64_000


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

    def test_minimax_cn_provider_preserves_dots(self):
        from types import SimpleNamespace
        agent = SimpleNamespace(provider="minimax-cn", base_url="")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_minimax_url_preserves_dots(self):
        from types import SimpleNamespace
        agent = SimpleNamespace(provider="custom", base_url="https://api.minimax.io/anthropic")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_minimax_cn_url_preserves_dots(self):
        from types import SimpleNamespace
        agent = SimpleNamespace(provider="custom", base_url="https://api.minimaxi.com/anthropic")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_anthropic_does_not_preserve_dots(self):
        from types import SimpleNamespace
        agent = SimpleNamespace(provider="anthropic", base_url="https://api.anthropic.com")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is False

    def test_opencode_zen_provider_preserves_dots(self):
        from types import SimpleNamespace
        agent = SimpleNamespace(provider="opencode-zen", base_url="")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_opencode_zen_url_preserves_dots(self):
        from types import SimpleNamespace
        agent = SimpleNamespace(provider="custom", base_url="https://opencode.ai/zen/v1")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_zai_provider_preserves_dots(self):
        from types import SimpleNamespace
        agent = SimpleNamespace(provider="zai", base_url="")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_bigmodel_cn_url_preserves_dots(self):
        from types import SimpleNamespace
        agent = SimpleNamespace(provider="custom", base_url="https://open.bigmodel.cn/api/paas/v4")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_normalize_preserves_m25_free_dot(self):
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name("minimax-m2.5-free", preserve_dots=True) == "minimax-m2.5-free"

    def test_normalize_preserves_m27_dot(self):
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name("MiniMax-M2.7", preserve_dots=True) == "MiniMax-M2.7"

    def test_normalize_preserves_non_anthropic_dots_without_preserve(self):
        from agent.anthropic_adapter import normalize_model_name
        # Non-Anthropic model families use dots as canonical version separators;
        # only Claude/Anthropic names are hyphen-normalized by default.
        assert normalize_model_name("MiniMax-M2.7", preserve_dots=False) == "MiniMax-M2.7"

    def test_normalize_still_converts_claude_dots_without_preserve(self):
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name("claude-opus-4.6", preserve_dots=False) == "claude-opus-4-6"


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
             patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-leaked") as mock_resolve, \
             patch("agent.anthropic_adapter._is_oauth_token", return_value=False):

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
