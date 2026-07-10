"""Tests for agent/model_metadata.py — token estimation, context lengths,
probing, caching, and error parsing.

Coverage levels:
  Token estimation       — concrete value assertions, edge cases
  Context length lookup  — resolution order, fuzzy match, cache priority
  API metadata fetch     — caching, TTL, canonical slugs, stale fallback
  Probe tiers            — descending, boundaries, extreme inputs
  Error parsing          — OpenAI, Ollama, Anthropic, edge cases
  Persistent cache       — save/load, corruption, update, provider isolation
"""

import time

import pytest
import yaml
from unittest.mock import patch, MagicMock

from agent.model_metadata import (
    CONTEXT_PROBE_TIERS,
    DEFAULT_CONTEXT_LENGTHS,
    DEFAULT_FALLBACK_CONTEXT,
    _strip_provider_prefix,
    estimate_tokens_rough,
    estimate_messages_tokens_rough,
    get_model_context_length,
    get_next_probe_tier,
    get_cached_context_length,
    parse_context_limit_from_error,
    save_context_length,
    fetch_model_metadata,
    _MODEL_CACHE_TTL,
    estimate_request_tokens_rough,
)


# =========================================================================
# Token estimation
# =========================================================================

class TestEstimateTokensRough:
    def test_empty_string(self):
        assert estimate_tokens_rough("") == 0

    def test_none_returns_zero(self):
        assert estimate_tokens_rough(None) == 0

    def test_known_length(self):
        assert estimate_tokens_rough("a" * 400) == 100

    def test_short_text(self):
        # "hello" = 5 chars → ceil(5/4) = 2
        assert estimate_tokens_rough("hello") == 2

    def test_proportional(self):
        short = estimate_tokens_rough("hello world")
        long = estimate_tokens_rough("hello world " * 100)
        assert long > short

    def test_unicode_multibyte(self):
        """Unicode chars are still 1 Python char each — 4 chars/token holds."""
        text = "你好世界"  # 4 CJK characters
        assert estimate_tokens_rough(text) == 1


class TestEstimateMessagesTokensRough:
    def test_empty_list(self):
        assert estimate_messages_tokens_rough([]) == 0

    def test_single_message_concrete_value(self):
        """Verify against known str(msg) length (ceiling division)."""
        msg = {"role": "user", "content": "a" * 400}
        result = estimate_messages_tokens_rough([msg])
        n = len(str(msg))
        expected = (n + 3) // 4
        assert result == expected

    def test_multiple_messages_additive(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there, how can I help?"},
        ]
        result = estimate_messages_tokens_rough(msgs)
        n = sum(len(str(m)) for m in msgs)
        expected = (n + 3) // 4
        assert result == expected

    def test_tool_call_message(self):
        """Tool call messages with no 'content' key still contribute tokens."""
        msg = {"role": "assistant", "content": None,
               "tool_calls": [{"id": "1", "function": {"name": "terminal", "arguments": "{}"}}]}
        result = estimate_messages_tokens_rough([msg])
        assert result > 0
        assert result == (len(str(msg)) + 3) // 4

    def test_message_with_list_content(self):
        """Vision messages with multimodal content arrays.

        Image parts are counted at a flat ~1500-token rate per image
        rather than counting the base64 char length, so a tiny stub
        payload still registers as full image cost.
        """
        msg = {"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ]}
        result = estimate_messages_tokens_rough([msg])
        # Flat cost = 1500 per image plus the small text overhead. Allow
        # a small band so this isn't a change-detector for the exact
        # string representation.
        assert 1500 <= result < 2000

    def test_message_with_huge_base64_image_stays_bounded(self):
        """A 1MB base64 PNG must not explode to ~250K tokens."""
        huge = "A" * (1024 * 1024)
        msg = {"role": "tool", "tool_call_id": "c1", "content": [
            {"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{huge}"}},
        ]}
        result = estimate_messages_tokens_rough([msg])
        assert result < 5000


class TestEstimateRequestTokensRough:
    def test_caches_tools_estimate(self):
        messages = [{"role": "user", "content": "hello"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run a command",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            }
        ]

        # json.dumps is used for params sizing; ensure the tools estimate is cached
        # so repeated calls don't keep re-serializing the same schema list.
        with patch("agent.model_metadata.json.dumps", wraps=__import__("json").dumps) as dumps:
            estimate_request_tokens_rough(messages, system_prompt="x" * 8, tools=tools)
            estimate_request_tokens_rough(messages, system_prompt="x" * 8, tools=tools)
            assert dumps.call_count == 1

    def test_tools_cache_is_bounded(self):
        # A long-lived process builds many transient tool lists; the cache must
        # not grow without bound. Feed more distinct lists than the cap and
        # confirm the cache never exceeds it.
        import agent.model_metadata as mm

        mm._TOOLS_TOKENS_CACHE.clear()
        cap = mm._TOOLS_TOKENS_CACHE_MAX
        # Keep references so ids are not recycled mid-loop, forcing distinct keys.
        held = []
        for i in range(cap + 50):
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": f"tool_{i}",
                        "description": "d",
                        "parameters": {"type": "object"},
                    },
                }
            ]
            held.append(tools)
            mm._estimate_tools_tokens_rough(tools)
            assert len(mm._TOOLS_TOKENS_CACHE) <= cap
        assert len(mm._TOOLS_TOKENS_CACHE) == cap


# =========================================================================
# Default context lengths
# =========================================================================

class TestDefaultContextLengths:
    def test_grok_substring_matching(self):
        # Longest-first substring matching must resolve the real xAI model
        # IDs to the correct fallback entries without 128k probe-down.
        from agent.model_metadata import get_model_context_length
        from unittest.mock import patch as mock_patch

        # Fake the provider/API/cache layers so the lookup falls through
        # to DEFAULT_CONTEXT_LENGTHS.
        with mock_patch("agent.model_metadata.fetch_model_metadata", return_value={}),              mock_patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}),              mock_patch("agent.model_metadata.get_cached_context_length", return_value=None):
            cases = [
                ("grok-4.20-0309-reasoning", 2000000),
                ("grok-4.20-0309-non-reasoning", 2000000),
                ("grok-4.20-multi-agent-0309", 2000000),
                ("grok-4-fast-reasoning", 2000000),
                ("grok-4-fast-non-reasoning", 2000000),
                ("grok-4", 256000),
                ("grok-4-0709", 256000),
                ("grok-build-0.1", 256000),
                ("grok-composer-2.5-fast", 200000),
                ("grok-code-fast-1", 256000),
                ("grok-3", 131072),
                ("grok-3-mini", 131072),
                ("grok-3-mini-fast", 131072),
                ("grok-2", 131072),
                ("grok-2-vision", 8192),
                ("grok-2-vision-1212", 8192),
                ("grok-beta", 131072),
            ]
            for model_id, expected_ctx in cases:
                actual = get_model_context_length(model_id)
                assert actual == expected_ctx, (
                    f"{model_id}: expected {expected_ctx}, got {actual}"
                )

    def test_xai_oauth_grok_build_uses_xai_models_dev_context(self):
        """xAI OAuth should share the xAI provider metadata path.

        The xAI /v1/models endpoint does not currently include context fields
        for grok-build-0.1, so this guards against falling through to the
        generic "grok" 131k fallback when using OAuth credentials.
        """
        registry = {
            "xai": {
                "models": {
                    "grok-build-0.1": {
                        "limit": {"context": 256000, "output": 64000},
                    },
                },
            },
        }
        with patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata._query_ollama_api_show", return_value=None), \
             patch("agent.models_dev.fetch_models_dev", return_value=registry):
            assert get_model_context_length(
                "grok-build-0.1",
                provider="xai-oauth",
                base_url="https://api.x.ai/v1",
                api_key="oauth-token",
            ) == 256000

    def test_deepseek_v4_models_1m_context(self):
        from agent.model_metadata import get_model_context_length
        from unittest.mock import patch as mock_patch

        expected_keys = {
            "deepseek-v4-pro": 1_000_000,
            "deepseek-v4-flash": 1_000_000,
            "deepseek-chat": 1_000_000,
            "deepseek-reasoner": 1_000_000,
        }
        for key, value in expected_keys.items():
            assert key in DEFAULT_CONTEXT_LENGTHS, f"{key} missing"
            assert DEFAULT_CONTEXT_LENGTHS[key] == value, (
                f"{key} should be {value}, got {DEFAULT_CONTEXT_LENGTHS[key]}"
            )

        # Longest-first substring matching must resolve both the bare V4
        # ids (native DeepSeek) and the vendor-prefixed forms (OpenRouter
        # / Nous Portal) to 1M without probing down to the legacy 128K
        # ``deepseek`` substring fallback.
        with mock_patch("agent.model_metadata.fetch_model_metadata", return_value={}), \
             mock_patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}), \
             mock_patch("agent.model_metadata.get_cached_context_length", return_value=None):
            cases = [
                ("deepseek-v4-pro", 1_000_000),
                ("deepseek-v4-flash", 1_000_000),
                ("deepseek/deepseek-v4-pro", 1_000_000),
                ("deepseek/deepseek-v4-flash", 1_000_000),
                ("deepseek-chat", 1_000_000),
                ("deepseek-reasoner", 1_000_000),
            ]
            for model_id, expected_ctx in cases:
                actual = get_model_context_length(model_id)
                assert actual == expected_ctx, (
                    f"{model_id}: expected {expected_ctx}, got {actual}"
                )

    def test_glm_52_context_1m(self):
        """GLM-5.2 must resolve to 1M, not the generic GLM fallback of 202K.

        Context window was verified empirically via needle-in-a-haystack
        retrieval at 789K prompt tokens on api.z.ai/api/coding/paas/v4
        (2026-06-13).
        """
        from agent.model_metadata import get_model_context_length
        from unittest.mock import patch as mock_patch

        assert DEFAULT_CONTEXT_LENGTHS["glm-5.2"] == 1_048_576
        assert DEFAULT_CONTEXT_LENGTHS["glm"] == 202752

        with mock_patch("agent.model_metadata.fetch_model_metadata", return_value={}), \
             mock_patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}), \
             mock_patch("agent.model_metadata.get_cached_context_length", return_value=None):
            # GLM-5.2 (1M) must NOT fall through to the generic 202K entry
            assert get_model_context_length("glm-5.2") == 1_048_576
            # Vendor-prefixed forms (zai provider, zhipu alias)
            assert get_model_context_length("zai/glm-5.2") == 1_048_576
            assert get_model_context_length("zhipu/glm-5.2") == 1_048_576
            # Older GLM variants still resolve to the generic 202K fallback
            assert get_model_context_length("glm-5") == 202752
            assert get_model_context_length("glm-5.1") == 202752

    def test_openrouter_live_metadata_beats_hardcoded_catchall(self):
        """OpenRouter-routed slugs resolve via the live OR catalog before the
        hardcoded family catch-all.

        Regression for the claude-fable-5 under-report: a brand-new Anthropic
        slug that is absent from models.dev but present in OpenRouter's live
        catalog (with a 1M window) used to fall through to the generic
        ``"claude": 200000`` entry, because the step-6 OR fallback was gated on
        ``not effective_provider`` and ``effective_provider`` is "openrouter"
        for any OpenRouter selection. The dedicated step-5 OR branch must read
        the live value instead.
        """
        from agent.model_metadata import get_model_context_length
        from unittest.mock import patch as mock_patch

        or_url = "https://openrouter.ai/api/v1"
        live = {
            "anthropic/claude-fable-5": {"context_length": 1_000_000},
            "anthropic/claude-haiku-4.5": {"context_length": 200_000},
        }
        with mock_patch("agent.model_metadata.fetch_model_metadata", return_value=live), \
             mock_patch("agent.model_metadata._query_ollama_api_show", return_value=None), \
             mock_patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             mock_patch("agent.models_dev.lookup_models_dev_context", return_value=None):
            # The bug: would have returned 200_000 via the "claude" catch-all.
            assert get_model_context_length(
                "anthropic/claude-fable-5", base_url=or_url, provider="openrouter"
            ) == 1_000_000
            # A genuinely-200k model still resolves to its real OR value — the
            # fix reads per-model context, it does not blanket-bump to 1M.
            assert get_model_context_length(
                "anthropic/claude-haiku-4.5", base_url=or_url, provider="openrouter"
            ) == 200_000

    def test_openrouter_kimi_32k_underreport_still_guarded(self):
        """The live OR branch keeps the Kimi-family 32k underreport guard:
        a bogus 32768 from OpenRouter for a Kimi slug must NOT win — it falls
        through to the hardcoded default instead.
        """
        from agent.model_metadata import get_model_context_length
        from unittest.mock import patch as mock_patch

        or_url = "https://openrouter.ai/api/v1"
        live = {"moonshotai/kimi-k2.6": {"context_length": 32768}}
        with mock_patch("agent.model_metadata.fetch_model_metadata", return_value=live), \
             mock_patch("agent.model_metadata._query_ollama_api_show", return_value=None), \
             mock_patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             mock_patch("agent.models_dev.lookup_models_dev_context", return_value=None):
            ctx = get_model_context_length(
                "moonshotai/kimi-k2.6", base_url=or_url, provider="openrouter"
            )
            assert ctx != 32768, "Kimi 32k OR underreport must not be accepted"


# =========================================================================
# Codex OAuth context-window resolution (provider="openai-codex")
# =========================================================================

class TestCodexOAuthContextLength:
    """ChatGPT Codex OAuth context windows come from the authenticated
    /models catalogue and may differ from the static fallback table or the
    direct OpenAI API allocation. The fallback values below are conservative
    defaults used only when the live probe is unavailable.
    """

    def setup_method(self):
        import agent.model_metadata as mm
        mm._codex_oauth_context_cache = {}
        mm._codex_oauth_context_cache_time = 0.0

    def test_fallback_table_used_without_token(self):
        """With no access token, the hardcoded Codex fallback table wins
        over models.dev (which reports 1.05M for gpt-5.5 but Codex is 272k).
        """
        from agent.model_metadata import get_model_context_length

        expected = {
            "gpt-5.5": 272_000,
            "gpt-5.4": 272_000,
            "gpt-5.4-mini": 272_000,
            "gpt-5.3-codex": 272_000,
            "gpt-5.3-codex-spark": 128_000,
            "gpt-5.2-codex": 272_000,
            "gpt-5.1-codex-max": 272_000,
            "gpt-5.1-codex-mini": 272_000,
        }

        with patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata.save_context_length"):
            for model, expected_ctx in expected.items():
                ctx = get_model_context_length(
                    model=model,
                    base_url="https://chatgpt.com/backend-api/codex",
                    api_key="",
                    provider="openai-codex",
                )
                assert ctx == expected_ctx, (
                    f"Codex {model}: expected {expected_ctx} fallback, got {ctx} "
                    "(models.dev leakage?)"
                )

    def test_live_probe_overrides_fallback(self):
        """When a token is provided, the live /models probe is preferred
        and its context_window drives the result."""
        from agent.model_metadata import get_model_context_length

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "models": [
                {"slug": "gpt-5.5", "context_window": 300_000},
                {"slug": "gpt-5.4", "context_window": 400_000},
            ]
        }

        with patch("agent.model_metadata.requests.get", return_value=fake_response), \
             patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata.save_context_length"):
            ctx_55 = get_model_context_length(
                model="gpt-5.5",
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="fake-token",
                provider="openai-codex",
            )
            ctx_54 = get_model_context_length(
                model="gpt-5.4",
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="fake-token",
                provider="openai-codex",
            )
        assert ctx_55 == 300_000
        assert ctx_54 == 400_000

    def test_probe_failure_falls_back_to_hardcoded(self):
        """If the probe fails (non-200 / network error), we still return
        the hardcoded 272k rather than leaking through to models.dev 1.05M."""
        from agent.model_metadata import get_model_context_length

        fake_response = MagicMock()
        fake_response.status_code = 401
        fake_response.json.return_value = {}

        with patch("agent.model_metadata.requests.get", return_value=fake_response), \
             patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata.save_context_length"):
            ctx = get_model_context_length(
                model="gpt-5.5",
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="expired-token",
                provider="openai-codex",
            )
        assert ctx == 272_000

    def test_non_codex_providers_unaffected(self):
        """Resolving gpt-5.5 on non-Codex providers must NOT use the Codex
        272k override — OpenRouter / direct OpenAI API have different limits.
        """
        from agent.model_metadata import get_model_context_length

        # OpenRouter — should hit its own catalog path first; when mocked
        # empty, falls through to hardcoded DEFAULT_CONTEXT_LENGTHS (1.05M,
        # matching the real direct-API value — Codex OAuth's 272k cap is
        # provider-specific and must not leak here).
        with patch("agent.model_metadata.fetch_model_metadata", return_value={}), \
             patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}), \
             patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.models_dev.lookup_models_dev_context", return_value=None):
            ctx = get_model_context_length(
                model="openai/gpt-5.5",
                base_url="https://openrouter.ai/api/v1",
                api_key="",
                provider="openrouter",
            )
        assert ctx == 1_050_000, (
            f"Non-Codex gpt-5.5 resolved to {ctx}; Codex 272k override "
            "leaked outside openai-codex provider"
        )

    def test_stale_codex_cache_is_bypassed_and_live_probe_wins(self, tmp_path, monkeypatch):
        """A stale Codex disk entry must not mask the authenticated catalogue."""
        from agent import model_metadata as mm

        cache_file = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_file)

        base_url = "https://chatgpt.com/backend-api/codex"
        stale_key = f"gpt-5.6-terra@{base_url}"
        other_key = "other-model@https://api.openai.com/v1/"
        import yaml as _yaml
        cache_file.write_text(_yaml.dump({"context_lengths": {
            stale_key: 272_000,
            other_key: 128_000,
        }}))

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "models": [{"slug": "gpt-5.6-terra", "context_window": 372_000}]
        }

        with patch("agent.model_metadata.requests.get", return_value=fake_response) as mock_get:
            ctx = mm.get_model_context_length(
                model="gpt-5.6-terra",
                base_url=base_url,
                api_key="fake-token",
                provider="openai-codex",
            )

        assert ctx == 372_000
        mock_get.assert_called_once()
        remaining = _yaml.safe_load(cache_file.read_text()).get("context_lengths", {})
        assert remaining.get(stale_key) == 372_000
        assert remaining.get(other_key) == 128_000

    def test_codex_fallback_is_not_persisted(self, tmp_path, monkeypatch):
        """A failed live probe must not poison the persistent cache."""
        from agent import model_metadata as mm

        cache_file = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_file)
        base_url = "https://chatgpt.com/backend-api/codex"

        fake_response = MagicMock()
        fake_response.status_code = 401
        fake_response.json.return_value = {}

        with patch("agent.model_metadata.requests.get", return_value=fake_response), \
             patch("agent.model_metadata.save_context_length") as mock_save:
            ctx = mm.get_model_context_length(
                model="gpt-5.6-terra",
                base_url=base_url,
                api_key="expired-token",
                provider="openai-codex",
            )

        assert ctx == 272_000
        mock_save.assert_not_called()
        assert not cache_file.exists()

    def test_codex_cache_is_not_used_when_probe_fails(self, tmp_path, monkeypatch):
        """Even a previously live-looking Codex row must not suppress probing."""
        from agent import model_metadata as mm

        cache_file = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_file)
        base_url = "https://chatgpt.com/backend-api/codex"
        import yaml as _yaml
        cache_file.write_text(_yaml.dump({"context_lengths": {
            f"gpt-5.6-terra@{base_url}": 372_000,
        }}))

        fake_response = MagicMock()
        fake_response.status_code = 401
        fake_response.json.return_value = {}

        with patch("agent.model_metadata.requests.get", return_value=fake_response) as mock_get:
            ctx = mm.get_model_context_length(
                model="gpt-5.6-terra",
                base_url=base_url,
                api_key="expired-token",
                provider="openai-codex",
            )

        assert ctx == 272_000
        mock_get.assert_called_once()
        remaining = _yaml.safe_load(cache_file.read_text()).get("context_lengths", {})
        assert remaining.get(f"gpt-5.6-terra@{base_url}") == 372_000


# =========================================================================
# Nous Portal context-window resolution (provider="nous")
# =========================================================================

class TestNousPortalContextResolution:
    """Nous Portal /v1/models is authoritative for what Nous infra enforces
    and may diverge from the OpenRouter catalog.

    Invariants this class pins down:
      1. Portal value wins over the OR fallback.
      2. Portal-derived values are persisted to disk.
      3. OR-fallback values are NEVER persisted — otherwise a single portal
         blip would freeze the wrong value in via step-1 cache short-circuit.
      4. Pre-fix persistent-cache entries (seeded from the OR catalog) are
         bypassed at step 1 and overwritten once the portal responds.
      5. Pre-fix persistent-cache entries SURVIVE on disk when the portal
         is unreachable — no opportunistic invalidation that loses the only
         value we have.
    """

    def setup_method(self):
        import agent.model_metadata as mm
        mm._endpoint_model_metadata_cache.clear()
        mm._endpoint_model_metadata_cache_time.clear()

    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    @patch("agent.model_metadata.fetch_model_metadata")
    def test_portal_value_wins_over_openrouter_catalog(
        self, mock_or, mock_portal, tmp_path, monkeypatch
    ):
        """The motivating case: OR catalog says 1M for qwen3.6-plus, but
        the Nous portal correctly enforces 262144.  Portal must win."""
        import agent.model_metadata as mm
        cache_file = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_file)

        mock_portal.return_value = {
            "qwen3.6-plus": {"context_length": 262_144},
        }
        mock_or.return_value = {
            "qwen/qwen3.6-plus": {"context_length": 1_000_000},
        }

        ctx = mm.get_model_context_length(
            model="qwen3.6-plus",
            base_url="https://inference-api.nousresearch.com/v1",
            api_key="fake-token",
            provider="nous",
        )
        assert ctx == 262_144, (
            f"Portal must override OR catalog; got {ctx} (OR leak?)"
        )

    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    @patch("agent.model_metadata.fetch_model_metadata")
    def test_portal_value_is_persisted_to_disk(
        self, mock_or, mock_portal, tmp_path, monkeypatch
    ):
        """Portal-derived value should land in the persistent cache so
        cross-process callers (e.g. child agents) see the same value."""
        import agent.model_metadata as mm
        cache_file = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_file)

        mock_portal.return_value = {
            "qwen3.6-plus": {"context_length": 262_144},
        }
        mock_or.return_value = {}

        base_url = "https://inference-api.nousresearch.com/v1"
        ctx = mm.get_model_context_length(
            model="qwen3.6-plus",
            base_url=base_url,
            api_key="fake",
            provider="nous",
        )
        assert ctx == 262_144
        persisted = yaml.safe_load(cache_file.read_text()).get("context_lengths", {})
        assert persisted.get(f"qwen3.6-plus@{base_url}") == 262_144, (
            "Portal-derived value should be persisted to disk"
        )

    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    @patch("agent.model_metadata.fetch_model_metadata")
    def test_openrouter_fallback_is_not_persisted(
        self, mock_or, mock_portal, tmp_path, monkeypatch
    ):
        """When the portal can't resolve a model (network blip, auth glitch,
        model not yet listed) we fall back to the OR catalog so the agent
        keeps working — but we must NOT write the OR value to disk.  Once
        cached on disk, step-1 short-circuits forever and the user is stuck
        with the wrong number until they manually clear the cache."""
        import agent.model_metadata as mm
        cache_file = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_file)

        mock_portal.return_value = {}  # portal unreachable / model unknown
        mock_or.return_value = {
            "qwen/qwen3.6-plus": {"context_length": 1_000_000},
        }

        base_url = "https://inference-api.nousresearch.com/v1"
        ctx = mm.get_model_context_length(
            model="qwen3.6-plus",
            base_url=base_url,
            api_key="fake",
            provider="nous",
        )
        assert ctx == 1_000_000, "OR fallback should still serve the request"
        assert not cache_file.exists() or not yaml.safe_load(
            cache_file.read_text()
        ).get("context_lengths", {}), (
            "OR-fallback values must NOT be persisted — a single portal blip "
            "would otherwise freeze the wrong value in via step-1 cache hit"
        )

    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    @patch("agent.model_metadata.fetch_model_metadata")
    def test_stale_cache_is_bypassed_and_overwritten_by_portal(
        self, mock_or, mock_portal, tmp_path, monkeypatch
    ):
        """Users upgrading from pre-fix builds have ``qwen3.6-plus@…nous… =
        1000000`` (OR-derived) sitting in their cache file.  Step 1 must
        NOT short-circuit on that entry — step 5b reconciles against the
        portal and overwrites the persistent value with 262144."""
        import agent.model_metadata as mm
        cache_file = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_file)

        base_url = "https://inference-api.nousresearch.com/v1"
        stale_key = f"qwen3.6-plus@{base_url}"
        other_key = "other-model@https://api.openai.com/v1"
        cache_file.write_text(yaml.dump({"context_lengths": {
            stale_key: 1_000_000,     # pre-fix OR-derived value
            other_key: 128_000,       # unrelated, must survive
        }}))

        mock_portal.return_value = {
            "qwen3.6-plus": {"context_length": 262_144},
        }
        mock_or.return_value = {}

        ctx = mm.get_model_context_length(
            model="qwen3.6-plus",
            base_url=base_url,
            api_key="fake",
            provider="nous",
        )
        assert ctx == 262_144, (
            f"Stale OR-derived cache entry should not have leaked through; got {ctx}"
        )

        remaining = yaml.safe_load(cache_file.read_text()).get("context_lengths", {})
        assert remaining.get(stale_key) == 262_144, (
            "Portal value should have overwritten the stale entry on disk"
        )
        assert remaining.get(other_key) == 128_000, (
            "Unrelated cache entries must not be touched"
        )

    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    @patch("agent.model_metadata.fetch_model_metadata")
    def test_stale_cache_survives_when_portal_unreachable(
        self, mock_or, mock_portal, tmp_path, monkeypatch
    ):
        """When the portal is unreachable AND we have a (potentially stale)
        on-disk cache entry, the entry must survive untouched — we don't
        want a transient outage to delete the only value we have.  The
        request itself still gets served via OR fallback for this call."""
        import agent.model_metadata as mm
        cache_file = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_file)

        base_url = "https://inference-api.nousresearch.com/v1"
        existing_key = f"qwen3.6-plus@{base_url}"
        cache_file.write_text(yaml.dump({"context_lengths": {
            existing_key: 1_000_000,
        }}))

        mock_portal.return_value = {}  # portal unreachable
        mock_or.return_value = {
            "qwen/qwen3.6-plus": {"context_length": 1_000_000},
        }

        mm.get_model_context_length(
            model="qwen3.6-plus",
            base_url=base_url,
            api_key="fake",
            provider="nous",
        )

        remaining = yaml.safe_load(cache_file.read_text()).get("context_lengths", {})
        assert remaining.get(existing_key) == 1_000_000, (
            "Persistent cache entry must survive a transient portal outage"
        )

    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    @patch("agent.model_metadata.fetch_model_metadata")
    def test_bypass_keyed_on_url_not_provider_string(
        self, mock_or, mock_portal, tmp_path, monkeypatch
    ):
        """Some call sites pass ``provider=""`` or ``provider="openrouter"``
        when the user is really on Nous Portal (e.g. cred-pool fallback).
        The Nous-URL bypass must trigger off the URL host, not the provider
        string, so the portal-first resolver still runs in that case."""
        import agent.model_metadata as mm
        cache_file = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_file)

        base_url = "https://inference-api.nousresearch.com/v1"
        cache_file.write_text(yaml.dump({"context_lengths": {
            f"qwen3.6-plus@{base_url}": 1_000_000,  # stale
        }}))

        mock_portal.return_value = {
            "qwen3.6-plus": {"context_length": 262_144},
        }
        mock_or.return_value = {}

        for provider_arg in ("", "openrouter", "custom"):
            mm._endpoint_model_metadata_cache.clear()
            mm._endpoint_model_metadata_cache_time.clear()
            ctx = mm.get_model_context_length(
                model="qwen3.6-plus",
                base_url=base_url,
                api_key="fake",
                provider=provider_arg,
            )
            assert ctx == 262_144, (
                f"URL-based Nous detection must fire for provider={provider_arg!r}; "
                f"got {ctx}"
            )


# =========================================================================
# get_model_context_length — resolution order
# =========================================================================

class TestGetModelContextLength:
    @patch("agent.model_metadata.fetch_model_metadata")
    def test_known_model_from_api(self, mock_fetch):
        mock_fetch.return_value = {
            "test/model": {"context_length": 32000}
        }
        assert get_model_context_length("test/model") == 32000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_fallback_to_defaults(self, mock_fetch):
        mock_fetch.return_value = {}
        assert get_model_context_length("anthropic/claude-sonnet-4") == 200000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_unknown_model_returns_first_probe_tier(self, mock_fetch):
        mock_fetch.return_value = {}
        assert get_model_context_length("unknown/never-heard-of-this") == CONTEXT_PROBE_TIERS[0]

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_partial_match_in_defaults(self, mock_fetch):
        mock_fetch.return_value = {}
        assert get_model_context_length("openai/gpt-4o") == 128000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_qwen3_coder_plus_context_length(self, mock_fetch):
        """qwen3-coder-plus has a 1M context window, not the generic 128K Qwen default."""
        mock_fetch.return_value = {}
        assert get_model_context_length("qwen3-coder-plus") == 1000000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_qwen3_coder_context_length(self, mock_fetch):
        """qwen3-coder has a 256K context window, not the generic 128K Qwen default."""
        mock_fetch.return_value = {}
        assert get_model_context_length("qwen3-coder") == 262144

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_qwen3_6_plus_context_length(self, mock_fetch):
        """qwen3.6-plus has a 1M context window, not the generic 128K Qwen default."""
        mock_fetch.return_value = {}
        assert get_model_context_length("qwen3.6-plus") == 1048576
        # Provider-prefixed variants must resolve to the same explicit entry
        # via the longest-substring fallback (no portal/OR cache available).
        assert get_model_context_length("qwen/qwen3.6-plus") == 1048576
        assert get_model_context_length("dashscope/qwen3.6-plus") == 1048576

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_qwen_generic_context_length(self, mock_fetch):
        """Generic qwen models still get the 128K default."""
        mock_fetch.return_value = {}
        assert get_model_context_length("qwen3-plus") == 131072

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_api_missing_context_length_key(self, mock_fetch):
        """Model in API but without context_length → defaults to the top
        probe tier (currently 256K)."""
        mock_fetch.return_value = {"test/model": {"name": "Test"}}
        assert get_model_context_length("test/model") == CONTEXT_PROBE_TIERS[0]

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_cache_takes_priority_over_api(self, mock_fetch, tmp_path):
        """Persistent cache should be checked BEFORE API metadata."""
        mock_fetch.return_value = {"my/model": {"context_length": 999999}}
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("my/model", "http://local", 32768)
            result = get_model_context_length("my/model", base_url="http://local")
            assert result == 32768  # cache wins over API's 999999

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_no_base_url_skips_cache(self, mock_fetch, tmp_path):
        """Without base_url, cache lookup is skipped."""
        mock_fetch.return_value = {}
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("custom/model", "http://local", 32768)
            # No base_url → cache skipped → falls to probe tier
            result = get_model_context_length("custom/model")
            assert result == CONTEXT_PROBE_TIERS[0]

    @patch("agent.model_metadata.fetch_model_metadata")
    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    def test_custom_endpoint_metadata_beats_fuzzy_default(self, mock_endpoint_fetch, mock_fetch):
        mock_fetch.return_value = {}
        mock_endpoint_fetch.return_value = {
            "zai-org/GLM-5-TEE": {"context_length": 65536}
        }

        result = get_model_context_length(
            "zai-org/GLM-5-TEE",
            base_url="https://llm.chutes.ai/v1",
            api_key="test-key",
        )

        assert result == 65536

    @patch("agent.model_metadata.fetch_model_metadata")
    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    def test_custom_endpoint_without_metadata_falls_back_to_catalog(self, mock_endpoint_fetch, mock_fetch):
        """Custom endpoint with no metadata should fall back to the hardcoded
        catalog (not 256K) when the model name matches a known entry.

        Previously this returned CONTEXT_PROBE_TIERS[0] (256K) because the
        custom-endpoint branch short-circuited before the catalog lookup.
        See #38865.
        """
        mock_fetch.return_value = {}
        mock_endpoint_fetch.return_value = {}

        # GLM-5-TEE matches the "glm" entry in DEFAULT_CONTEXT_LENGTHS
        result = get_model_context_length(
            "zai-org/GLM-5-TEE",
            base_url="https://llm.chutes.ai/v1",
            api_key="test-key",
        )
        assert result == 202752  # "glm" entry in DEFAULT_CONTEXT_LENGTHS

    @patch("agent.model_metadata.fetch_model_metadata")
    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    def test_custom_endpoint_single_model_fallback(self, mock_endpoint_fetch, mock_fetch):
        """Single-model servers: use the only model even if name doesn't match."""
        mock_fetch.return_value = {}
        mock_endpoint_fetch.return_value = {
            "Qwen3.5-9B-Q4_K_M.gguf": {"context_length": 131072}
        }

        result = get_model_context_length(
            "qwen3.5:9b",
            base_url="http://myserver.example.com:8080/v1",
            api_key="test-key",
        )

        assert result == 131072

    @patch("agent.model_metadata.fetch_model_metadata")
    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    def test_custom_endpoint_fuzzy_substring_match(self, mock_endpoint_fetch, mock_fetch):
        """Fuzzy match: configured model name is substring of endpoint model."""
        mock_fetch.return_value = {}
        mock_endpoint_fetch.return_value = {
            "org/llama-3.3-70b-instruct-fp8": {"context_length": 131072},
            "org/qwen-2.5-72b": {"context_length": 32768},
        }

        result = get_model_context_length(
            "llama-3.3-70b-instruct",
            base_url="http://myserver.example.com:8080/v1",
            api_key="test-key",
        )

        assert result == 131072

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_config_context_length_overrides_all(self, mock_fetch):
        """Explicit config_context_length takes priority over everything."""
        mock_fetch.return_value = {
            "test/model": {"context_length": 200000}
        }

        result = get_model_context_length(
            "test/model",
            config_context_length=65536,
        )

        assert result == 65536

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_config_context_length_zero_is_ignored(self, mock_fetch):
        """config_context_length=0 should be treated as unset."""
        mock_fetch.return_value = {}

        result = get_model_context_length(
            "anthropic/claude-sonnet-4",
            config_context_length=0,
        )

        assert result == 200000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_config_context_length_none_is_ignored(self, mock_fetch):
        """config_context_length=None should be treated as unset."""
        mock_fetch.return_value = {}

        result = get_model_context_length(
            "anthropic/claude-sonnet-4",
            config_context_length=None,
        )

        assert result == 200000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_custom_endpoint_falls_back_to_hardcoded_catalog(self, mock_fetch):
        """Custom/proxied endpoint that fails all probes should still resolve
        via DEFAULT_CONTEXT_LENGTHS instead of returning 256K.

        Regression test for #38865: a corporate Anthropic proxy (custom
        base_url) caused the custom-endpoint branch to short-circuit before
        the catalog lookup, capping context at 256K even for models like
        claude-opus-4-8 that are in the hardcoded catalog with 1M.
        """
        mock_fetch.return_value = {}

        # Patch all the probe functions that the custom-endpoint branch calls
        # so they all fail (return None/empty), simulating a proxy that
        # doesn't expose Ollama or local-server endpoints.
        with (
            patch(
                "agent.model_metadata._resolve_endpoint_context_length",
                return_value=None,
            ),
            patch(
                "agent.model_metadata._query_ollama_api_show",
                return_value=None,
            ),
            patch(
                "agent.model_metadata._query_local_context_length",
                return_value=None,
            ),
            patch(
                "agent.model_metadata.is_local_endpoint",
                return_value=False,
            ),
        ):
            # A known model behind a custom proxy should resolve to its
            # catalog value (1M), NOT the 256K fallback.
            ctx = get_model_context_length(
                "claude-opus-4-8",
                base_url="https://my-gateway.example.com/v1/claude",
            )
            assert ctx == 1000000, f"Expected 1000000, got {ctx}"

            # Another known model
            ctx2 = get_model_context_length(
                "claude-sonnet-4-6",
                base_url="https://my-gateway.example.com/v1/claude",
            )
            assert ctx2 == 1000000, f"Expected 1000000, got {ctx2}"

            # An unknown model on a custom endpoint should still fall back
            # to 256K (no catalog match).
            ctx3 = get_model_context_length(
                "totally-unknown-model",
                base_url="https://my-gateway.example.com/v1/claude",
            )
            assert ctx3 == DEFAULT_FALLBACK_CONTEXT, (
                f"Expected {DEFAULT_FALLBACK_CONTEXT}, got {ctx3}"
            )


# =========================================================================
# Bedrock context resolution — must run BEFORE custom-endpoint probe
# =========================================================================

class TestBedrockContextResolution:
    """Regression tests for Bedrock context-length resolution order.

    Bug: because ``bedrock-runtime.<region>.amazonaws.com`` is not listed in
    ``_URL_TO_PROVIDER``, ``_is_known_provider_base_url`` returned False and
    the custom-endpoint probe at step 2 ran first — fetching ``/models`` from
    Bedrock (which it doesn't serve), returning the 128K default-fallback
    before execution ever reached the Bedrock branch.

    Fix: promote the Bedrock branch ahead of the custom-endpoint probe.
    """

    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    def test_bedrock_provider_returns_static_table_before_probe(self, mock_fetch):
        """provider='bedrock' resolves via static table, bypasses /models probe."""
        ctx = get_model_context_length(
            "anthropic.claude-opus-4-v1:0",
            provider="bedrock",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        )
        # Must return the static Bedrock table value (200K for Claude),
        # NOT DEFAULT_FALLBACK_CONTEXT (128K).
        assert ctx == 200000
        mock_fetch.assert_not_called()

    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    def test_bedrock_url_without_provider_hint(self, mock_fetch):
        """bedrock-runtime host infers Bedrock even when provider is omitted."""
        ctx = get_model_context_length(
            "anthropic.claude-sonnet-4-v1:0",
            base_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        )
        assert ctx == 200000
        mock_fetch.assert_not_called()

    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    def test_non_bedrock_url_still_probes(self, mock_fetch):
        """Non-Bedrock hosts still reach the custom-endpoint probe."""
        mock_fetch.return_value = {"some-model": {"context_length": 50000}}
        ctx = get_model_context_length(
            "some-model",
            base_url="https://api.example.com/v1",
        )
        assert ctx == 50000
        assert mock_fetch.called


# =========================================================================
# _strip_provider_prefix — Ollama model:tag vs provider:model
# =========================================================================

class TestStripProviderPrefix:
    def test_known_provider_prefix_is_stripped(self):
        assert _strip_provider_prefix("local:my-model") == "my-model"
        assert _strip_provider_prefix("openrouter:anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"
        assert _strip_provider_prefix("anthropic:claude-sonnet-4") == "claude-sonnet-4"
        assert _strip_provider_prefix("stepfun:step-3.5-flash") == "step-3.5-flash"

    def test_ollama_model_tag_preserved(self):
        """Ollama model:tag format must NOT be stripped."""
        assert _strip_provider_prefix("qwen3.5:27b") == "qwen3.5:27b"
        assert _strip_provider_prefix("llama3.3:70b") == "llama3.3:70b"
        assert _strip_provider_prefix("gemma2:9b") == "gemma2:9b"
        assert _strip_provider_prefix("codellama:13b-instruct-q4_0") == "codellama:13b-instruct-q4_0"

    def test_http_urls_preserved(self):
        assert _strip_provider_prefix("http://example.com") == "http://example.com"
        assert _strip_provider_prefix("https://example.com") == "https://example.com"

    def test_no_colon_returns_unchanged(self):
        assert _strip_provider_prefix("gpt-4o") == "gpt-4o"
        assert _strip_provider_prefix("anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_ollama_model_tag_not_mangled_in_context_lookup(self, mock_fetch):
        """Ensure 'qwen3.5:27b' is NOT reduced to '27b' during context length lookup.

        We mock a custom endpoint that knows 'qwen3.5:27b' — the full name
        must reach the endpoint metadata lookup intact.
        """
        mock_fetch.return_value = {}
        with patch("agent.model_metadata.fetch_endpoint_model_metadata") as mock_ep, \
             patch("agent.model_metadata._is_custom_endpoint", return_value=True):
            mock_ep.return_value = {"qwen3.5:27b": {"context_length": 32768}}
            result = get_model_context_length(
                "qwen3.5:27b",
                base_url="http://localhost:11434/v1",
            )
        assert result == 32768


# =========================================================================
# fetch_model_metadata — caching, TTL, slugs, failures
# =========================================================================

class TestFetchModelMetadata:
    def _reset_cache(self):
        import agent.model_metadata as mm
        mm._model_metadata_cache = {}
        mm._model_metadata_cache_time = 0

    def _isolate_disk_cache(self, monkeypatch, tmp_path):
        import agent.model_metadata as mm
        cache_path = tmp_path / "openrouter_model_metadata.json"
        monkeypatch.setattr(mm, "_get_model_metadata_cache_path", lambda: cache_path)
        return cache_path

    def test_fresh_disk_cache_skips_network(self, tmp_path, monkeypatch):
        self._reset_cache()
        cache_path = self._isolate_disk_cache(monkeypatch, tmp_path)
        cache_path.write_text(
            '{"test/model":{"context_length":12345,"name":"Cached","pricing":{}}}',
            encoding="utf-8",
        )

        with patch("agent.model_metadata.requests.get") as mock_get:
            result = fetch_model_metadata()

        mock_get.assert_not_called()
        assert result["test/model"]["context_length"] == 12345

    def test_force_refresh_bypasses_fresh_disk_cache(self, tmp_path, monkeypatch):
        self._reset_cache()
        cache_path = self._isolate_disk_cache(monkeypatch, tmp_path)
        cache_path.write_text(
            '{"test/model":{"context_length":12345,"name":"Cached","pricing":{}}}',
            encoding="utf-8",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": "live/model", "context_length": 67890, "name": "Live"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("agent.model_metadata.requests.get", return_value=mock_response) as mock_get:
            result = fetch_model_metadata(force_refresh=True)

        assert mock_get.call_count == 1
        assert "live/model" in result
        assert "test/model" not in result

    def test_network_success_writes_disk_cache(self, tmp_path, monkeypatch):
        self._reset_cache()
        cache_path = self._isolate_disk_cache(monkeypatch, tmp_path)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": "live/model", "context_length": 67890, "name": "Live"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("agent.model_metadata.requests.get", return_value=mock_response):
            fetch_model_metadata(force_refresh=True)

        assert cache_path.exists()
        assert "live/model" in cache_path.read_text(encoding="utf-8")

    def test_network_failure_falls_back_to_stale_disk_cache(self, tmp_path, monkeypatch):
        self._reset_cache()
        cache_path = self._isolate_disk_cache(monkeypatch, tmp_path)
        cache_path.write_text(
            '{"stale/model":{"context_length":50000,"name":"Stale","pricing":{}}}',
            encoding="utf-8",
        )
        old = time.time() - _MODEL_CACHE_TTL - 60
        import os
        os.utime(cache_path, (old, old))

        with patch("agent.model_metadata.requests.get", side_effect=Exception("Network error")):
            result = fetch_model_metadata(force_refresh=True)

        assert result["stale/model"]["context_length"] == 50000

    @patch("agent.model_metadata.requests.get")
    def test_caches_result(self, mock_get):
        self._reset_cache()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": "test/model", "context_length": 99999, "name": "Test"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result1 = fetch_model_metadata(force_refresh=True)
        assert "test/model" in result1
        assert mock_get.call_count == 1

        result2 = fetch_model_metadata()
        assert "test/model" in result2
        assert mock_get.call_count == 1  # cached

    @patch("agent.model_metadata.requests.get")
    def test_api_failure_returns_empty_on_cold_cache(self, mock_get):
        self._reset_cache()
        mock_get.side_effect = Exception("Network error")
        result = fetch_model_metadata(force_refresh=True)
        assert result == {}

    @patch("agent.model_metadata.requests.get")
    def test_api_failure_returns_stale_cache(self, mock_get):
        """On API failure with existing cache, stale data is returned."""
        import agent.model_metadata as mm
        mm._model_metadata_cache = {"old/model": {"context_length": 50000}}
        mm._model_metadata_cache_time = 0  # expired

        mock_get.side_effect = Exception("Network error")
        result = fetch_model_metadata(force_refresh=True)
        assert "old/model" in result
        assert result["old/model"]["context_length"] == 50000

    @patch("agent.model_metadata.requests.get")
    def test_canonical_slug_aliasing(self, mock_get):
        """Models with canonical_slug get indexed under both IDs."""
        self._reset_cache()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{
                "id": "anthropic/claude-3.5-sonnet:beta",
                "canonical_slug": "anthropic/claude-3.5-sonnet",
                "context_length": 200000,
                "name": "Claude 3.5 Sonnet"
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_model_metadata(force_refresh=True)
        # Both the original ID and canonical slug should work
        assert "anthropic/claude-3.5-sonnet:beta" in result
        assert "anthropic/claude-3.5-sonnet" in result
        assert result["anthropic/claude-3.5-sonnet"]["context_length"] == 200000

    @patch("agent.model_metadata.requests.get")
    def test_provider_prefixed_models_get_bare_aliases(self, mock_get):
        self._reset_cache()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{
                "id": "provider/test-model",
                "context_length": 123456,
                "name": "Provider: Test Model",
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_model_metadata(force_refresh=True)

        assert result["provider/test-model"]["context_length"] == 123456
        assert result["test-model"]["context_length"] == 123456

    @patch("agent.model_metadata.requests.get")
    def test_ttl_expiry_triggers_refetch(self, mock_get, tmp_path, monkeypatch):
        """Cache expires after _MODEL_CACHE_TTL seconds."""
        import agent.model_metadata as mm
        self._reset_cache()
        cache_path = self._isolate_disk_cache(monkeypatch, tmp_path)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": "m1", "context_length": 1000, "name": "M1"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        fetch_model_metadata(force_refresh=True)
        assert mock_get.call_count == 1

        # Simulate both memory and disk TTL expiry.
        mm._model_metadata_cache_time = time.time() - _MODEL_CACHE_TTL - 1
        old = time.time() - _MODEL_CACHE_TTL - 1
        import os
        os.utime(cache_path, (old, old))
        fetch_model_metadata()
        assert mock_get.call_count == 2  # refetched

    @patch("agent.model_metadata.requests.get")
    def test_malformed_json_no_data_key(self, mock_get):
        """API returns JSON without 'data' key — empty cache, no crash."""
        self._reset_cache()
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "something"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_model_metadata(force_refresh=True)
        assert result == {}


# =========================================================================
# Context probe tiers
# =========================================================================

class TestContextProbeTiers:
    def test_tiers_descending(self):
        for i in range(len(CONTEXT_PROBE_TIERS) - 1):
            assert CONTEXT_PROBE_TIERS[i] > CONTEXT_PROBE_TIERS[i + 1]


class TestGetNextProbeTier:
    def test_from_256k(self):
        assert get_next_probe_tier(256_000) == 128_000

    def test_from_128k(self):
        assert get_next_probe_tier(128_000) == 64_000

    def test_from_64k(self):
        assert get_next_probe_tier(64_000) == 32_000

    def test_from_32k(self):
        assert get_next_probe_tier(32_000) == 16_000

    def test_from_8k_returns_none(self):
        assert get_next_probe_tier(8_000) is None

    def test_from_below_min_returns_none(self):
        assert get_next_probe_tier(4_000) is None

    def test_from_arbitrary_value(self):
        assert get_next_probe_tier(100_000) == 64_000

    def test_above_max_tier(self):
        """Value above 256K should return 256K."""
        assert get_next_probe_tier(500_000) == 256_000

    def test_zero_returns_none(self):
        assert get_next_probe_tier(0) is None


# =========================================================================
# Error message parsing
# =========================================================================

class TestParseContextLimitFromError:
    def test_openai_format(self):
        msg = "This model's maximum context length is 32768 tokens. However, your messages resulted in 45000 tokens."
        assert parse_context_limit_from_error(msg) == 32768

    def test_context_length_exceeded(self):
        msg = "context_length_exceeded: maximum context length is 131072"
        assert parse_context_limit_from_error(msg) == 131072

    def test_context_size_exceeded(self):
        msg = "Maximum context size 65536 exceeded"
        assert parse_context_limit_from_error(msg) == 65536

    def test_no_limit_in_message(self):
        assert parse_context_limit_from_error("Something went wrong with the API") is None

    def test_unreasonable_small_number_rejected(self):
        assert parse_context_limit_from_error("context length is 42 tokens") is None

    def test_ollama_format(self):
        msg = "Context size has been exceeded. Maximum context size is 32768"
        assert parse_context_limit_from_error(msg) == 32768

    def test_anthropic_format(self):
        msg = "prompt is too long: 250000 tokens > 200000 maximum"
        # Should extract 200000 (the limit), not 250000 (the input size)
        assert parse_context_limit_from_error(msg) == 200000

    def test_lmstudio_format(self):
        msg = "Error: context window of 4096 tokens exceeded"
        assert parse_context_limit_from_error(msg) == 4096

    def test_vllm_max_model_len_format(self):
        msg = (
            "The engine prompt length 1327246 exceeds the max_model_len 32768. "
            "Please reduce prompt."
        )
        assert parse_context_limit_from_error(msg) == 32768

    def test_vllm_maximum_model_length_format(self):
        msg = "prompt length 200000 exceeds maximum model length 131072"
        assert parse_context_limit_from_error(msg) == 131072

    @pytest.mark.parametrize("msg,expected", [
        ("max_model_len 32768", 32768),
        ("max_model_len: 32768", 32768),
        ("max_model_len=32768", 32768),
        ("max_model_len (32768)", 32768),
        ("max_model_len is 32768", 32768),
        ("maximum model length 131072", 131072),
        ("maximum model length is 131072", 131072),
        ("maximum model length: 131072", 131072),
    ])
    def test_vllm_delimiter_variants(self, msg, expected):
        """vLLM emits the limit with various delimiters (space/colon/equals/
        paren/'is'). The parser must catch all of them — the original
        space-only patterns silently missed ':', '=', '(' and 'is' forms and
        fell through to None."""
        assert parse_context_limit_from_error(msg) == expected

    def test_get_context_length_from_vllm_max_model_len_error(self):
        from agent.model_metadata import get_context_length_from_provider_error

        msg = (
            "The engine prompt length 90000 exceeds the max_model_len 32768. "
            "Please reduce prompt."
        )
        assert get_context_length_from_provider_error(msg, 131072) == 32768

    def test_minimax_delta_only_message_returns_none(self):
        msg = "invalid params, context window exceeds limit (2013)"
        assert parse_context_limit_from_error(msg) is None

    def test_completely_unrelated_error(self):
        assert parse_context_limit_from_error("Invalid API key") is None

    def test_empty_string(self):
        assert parse_context_limit_from_error("") is None

    def test_number_outside_reasonable_range(self):
        """Very large number (>10M) should be rejected."""
        msg = "maximum context length is 99999999999"
        assert parse_context_limit_from_error(msg) is None


# =========================================================================
# Persistent context length cache
# =========================================================================

class TestContextLengthCache:
    def test_save_and_load(self, tmp_path):
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("test/model", "http://localhost:8080/v1", 32768)
            assert get_cached_context_length("test/model", "http://localhost:8080/v1") == 32768

    def test_missing_cache_returns_none(self, tmp_path):
        cache_file = tmp_path / "nonexistent.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            assert get_cached_context_length("test/model", "http://x") is None

    def test_multiple_models_cached(self, tmp_path):
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("model-a", "http://a", 64000)
            save_context_length("model-b", "http://b", 128000)
            assert get_cached_context_length("model-a", "http://a") == 64000
            assert get_cached_context_length("model-b", "http://b") == 128000

    def test_same_model_different_providers(self, tmp_path):
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("llama-3", "http://local:8080", 32768)
            save_context_length("llama-3", "https://openrouter.ai/api/v1", 131072)
            assert get_cached_context_length("llama-3", "http://local:8080") == 32768
            assert get_cached_context_length("llama-3", "https://openrouter.ai/api/v1") == 131072

    def test_idempotent_save(self, tmp_path):
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("model", "http://x", 32768)
            save_context_length("model", "http://x", 32768)
            with open(cache_file) as f:
                data = yaml.safe_load(f)
            assert len(data["context_lengths"]) == 1

    def test_update_existing_value(self, tmp_path):
        """Saving a different value for the same key overwrites it."""
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("model", "http://x", 128000)
            save_context_length("model", "http://x", 64000)
            assert get_cached_context_length("model", "http://x") == 64000

    def test_corrupted_yaml_returns_empty(self, tmp_path):
        """Corrupted cache file is handled gracefully."""
        cache_file = tmp_path / "cache.yaml"
        cache_file.write_text("{{{{not valid yaml: [[[")
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            assert get_cached_context_length("model", "http://x") is None

    def test_wrong_structure_returns_none(self, tmp_path):
        """YAML that loads but has wrong structure."""
        cache_file = tmp_path / "cache.yaml"
        cache_file.write_text("just_a_string\n")
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            assert get_cached_context_length("model", "http://x") is None

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_cached_value_takes_priority(self, mock_fetch, tmp_path):
        mock_fetch.return_value = {}
        cache_file = tmp_path / "cache.yaml"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length("unknown/model", "http://local", 65536)
            assert get_model_context_length("unknown/model", base_url="http://local") == 65536

    def test_special_chars_in_model_name(self, tmp_path):
        """Model names with colons, slashes, etc. don't break the cache."""
        cache_file = tmp_path / "cache.yaml"
        model = "anthropic/claude-3.5-sonnet:beta"
        url = "https://api.example.com/v1"
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            save_context_length(model, url, 200000)
            assert get_cached_context_length(model, url) == 200000


class TestGrok43StaleCacheGuard:
    """Pre-catalog builds resolved grok-4.3 via the generic 'grok-4' catch-all
    (256,000) and persisted it before the 'grok-4.3' (1M) catalog entry was
    added on 2026-05-15.  The step-1 cache guard must drop that stale value
    and re-resolve to 1M, while leaving correct grok-4 entries (256,000)
    untouched.
    """

    def test_suggests_grok_4_3(self):
        from agent.model_metadata import _model_name_suggests_grok_4_3
        assert _model_name_suggests_grok_4_3("grok-4.3")
        assert _model_name_suggests_grok_4_3("grok-4.3-latest")
        assert _model_name_suggests_grok_4_3("xai/grok-4.3")
        assert not _model_name_suggests_grok_4_3("grok-4")
        assert not _model_name_suggests_grok_4_3("grok-4-fast")
        assert not _model_name_suggests_grok_4_3("grok-4.20")

    def test_stale_grok_4_3_dropped_and_reresolves_to_1m(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import importlib
        import agent.model_metadata as mm
        importlib.reload(mm)
        base = "https://api.x.ai/v1"
        mm.save_context_length("grok-4.3", base, 256_000)
        ctx = mm.get_model_context_length(
            "grok-4.3", base_url=base, api_key="", provider="xai"
        )
        assert ctx == 1_000_000

    def test_correct_grok_4_3_cache_preserved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import importlib
        import agent.model_metadata as mm
        importlib.reload(mm)
        base = "https://api.x.ai/v1"
        mm.save_context_length("grok-4.3", base, 1_000_000)
        ctx = mm.get_model_context_length(
            "grok-4.3", base_url=base, api_key="", provider="xai"
        )
        assert ctx == 1_000_000

    def test_grok_4_not_clobbered(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import importlib
        import agent.model_metadata as mm
        importlib.reload(mm)
        base = "https://api.x.ai/v1"
        # 256,000 is the CORRECT value for plain grok-4 — guard must not touch it.
        for slug in ("grok-4", "grok-4-0709"):
            mm.save_context_length(slug, base, 256_000)
            ctx = mm.get_model_context_length(
                slug, base_url=base, api_key="", provider="xai"
            )
            assert ctx == 256_000, f"{slug} should stay 256000, got {ctx}"


class TestMoAContextLength:
    """MoA virtual provider resolves context from the aggregator slot, not 256K default."""

    def _write_moa_config(self, home, aggregator):
        import os
        os.makedirs(home, exist_ok=True)
        with open(os.path.join(home, "config.yaml"), "w") as f:
            yaml.safe_dump(
                {
                    "moa": {
                        "default_preset": "p",
                        "presets": {
                            "p": {
                                "enabled": True,
                                "reference_models": [
                                    {"provider": "openrouter", "model": "openai/gpt-5.5"}
                                ],
                                "aggregator": aggregator,
                            }
                        },
                    }
                },
                f,
            )

    def test_moa_resolves_from_aggregator(self, tmp_path, monkeypatch):
        home = str(tmp_path / ".hermes")
        monkeypatch.setenv("HERMES_HOME", home)
        self._write_moa_config(home, {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"})

        # The MoA preset name + virtual base_url would otherwise fall through to
        # the 256K default; instead it mirrors the aggregator's real window.
        agg_ctx = get_model_context_length(
            "anthropic/claude-opus-4.8", base_url="https://openrouter.ai/api/v1", provider="openrouter"
        )
        moa_ctx = get_model_context_length("p", base_url="http://127.0.0.1/v1", provider="moa")
        assert moa_ctx == agg_ctx

    def test_moa_config_override_still_wins(self, tmp_path, monkeypatch):
        home = str(tmp_path / ".hermes")
        monkeypatch.setenv("HERMES_HOME", home)
        self._write_moa_config(home, {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"})
        ctx = get_model_context_length(
            "p", base_url="http://127.0.0.1/v1", provider="moa", config_context_length=500_000
        )
        assert ctx == 500_000
