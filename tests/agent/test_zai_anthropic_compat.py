"""Tests for Z.AI (Zhipu GLM) Anthropic-compatible endpoint compatibility.

Z.AI serves GLM models behind an Anthropic wire relay at ``api.z.ai/api/anthropic``
and ``open.bigmodel.cn/api/anthropic``. The relay has two quirks that Hermes'
adapter must work around:

1. **Beta rejection (1302):** requests carrying Claude-specific
   ``anthropic-beta`` headers are rejected with HTTP 429 ``rate_limit_error``
   code 1302. An A/B test (5 alternating calls, same key/payload/window)
   showed 5/5 success without the betas and 0/5 with them, all rejections
   returning in <0.5s — deterministic, not rate-limit noise. The adapter
   strips ALL betas for Z.AI endpoints.

2. **System prompt size limit (1305):** system prompts larger than ~2 KB are
   rejected with ``overloaded_error`` code 1305 (5/5 success at <1 KB, 0/5 at
   104 KB). Message content has no such limit (420 K tokens of user content
   works), so the adapter folds large system prompts into the first user
   message for Z.AI. Native Anthropic and other relays are unaffected.
"""
import pytest


# ---------------------------------------------------------------------------
# Endpoint detection
# ---------------------------------------------------------------------------

class TestIsZaiAnthropicEndpoint:
    def _f(self, url):
        from agent.anthropic_adapter import _is_zai_anthropic_endpoint
        return _is_zai_anthropic_endpoint(url)

    def test_zai_global_endpoint(self):
        assert self._f("https://api.z.ai/api/anthropic")

    def test_zai_global_with_trailing_slash(self):
        assert self._f("https://api.z.ai/api/anthropic/")

    def test_zai_bigmodel_cn_endpoint(self):
        assert self._f("https://open.bigmodel.cn/api/anthropic")

    def test_zai_mixed_case(self):
        assert self._f("https://API.Z.AI/api/Anthropic")

    def test_none_url(self):
        assert not self._f(None)

    def test_empty_url(self):
        assert not self._f("")

    def test_native_anthropic_not_matched(self):
        assert not self._f("https://api.anthropic.com")

    def test_minimax_not_matched(self):
        assert not self._f("https://api.minimax.io/anthropic")

    def test_kimi_not_matched(self):
        assert not self._f("https://api.moonshot.cn/anthropic")

    def test_openai_compat_not_matched(self):
        assert not self._f("https://api.z.ai/api/coding/paas/v4")

    def test_unrelated_zai_not_matched(self):
        # The OpenAI-compat Z.AI endpoints must not trigger the anthropic fix.
        assert not self._f("https://api.z.ai/api/paas/v4")


# ---------------------------------------------------------------------------
# Beta stripping
# ---------------------------------------------------------------------------

class TestZaiBetaStripping:
    def test_zai_strips_all_betas(self):
        from agent.anthropic_adapter import (
            _common_betas_for_base_url,
            _COMMON_BETAS,
            _TOOL_STREAMING_BETA,
        )
        # Sanity: _COMMON_BETAS is non-empty and includes the Claude betas.
        assert _COMMON_BETAS
        assert _TOOL_STREAMING_BETA in _COMMON_BETAS
        # Z.AI must strip ALL of them (not just a subset like MiniMax).
        result = _common_betas_for_base_url("https://api.z.ai/api/anthropic")
        assert result == []

    def test_zai_bigmodel_cn_strips_all_betas(self):
        from agent.anthropic_adapter import _common_betas_for_base_url
        assert _common_betas_for_base_url("https://open.bigmodel.cn/api/anthropic") == []

    def test_native_anthropic_keeps_betas(self):
        """Native Anthropic endpoints must keep ALL betas (caching/thinking)."""
        from agent.anthropic_adapter import (
            _common_betas_for_base_url,
            _COMMON_BETAS,
            _TOOL_STREAMING_BETA,
        )
        result = _common_betas_for_base_url("https://api.anthropic.com")
        assert _TOOL_STREAMING_BETA in result
        assert result == _COMMON_BETAS

    def test_minimax_still_strips_only_specific_betas(self):
        """MiniMax behaviour must be unchanged by the Z.AI branch."""
        from agent.anthropic_adapter import (
            _common_betas_for_base_url,
            _TOOL_STREAMING_BETA,
        )
        result = _common_betas_for_base_url("https://api.minimax.io/anthropic")
        assert _TOOL_STREAMING_BETA not in result
        assert len(result) > 0  # MiniMax keeps some betas; Z.AI strips all


# ---------------------------------------------------------------------------
# System prompt folding
# ---------------------------------------------------------------------------

def _build_kwargs(base_url, system=None, messages=None, **extra):
    """Thin wrapper around build_anthropic_kwargs with sensible defaults."""
    from agent.anthropic_adapter import build_anthropic_kwargs
    kwargs = dict(
        model="glm-5.2",
        messages=messages if messages is not None else [{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=128,
        reasoning_config=None,
        base_url=base_url,
        context_length=1000000,
    )
    if system is not None:
        # build_anthropic_kwargs extracts system from a leading system-role message
        kwargs["messages"] = [{"role": "system", "content": system}] + kwargs["messages"]
    kwargs.update(extra)
    return build_anthropic_kwargs(**kwargs)


class TestZaiSystemPromptFolding:
    BIG_SYSTEM = "You are Zephyr. " + ("Important rule. " * 5000)  # ~60 KB

    def test_zai_large_system_folded_into_messages(self):
        """Large system prompt on Z.AI must move into the first user message."""
        kw = _build_kwargs(
            base_url="https://api.z.ai/api/anthropic",
            system=self.BIG_SYSTEM,
            messages=[{"role": "user", "content": "hello"}],
        )
        # system field must NOT be present (folded instead)
        assert "system" not in kw
        # the system content must appear in the message stream, wrapped in <system>
        msg_blob = str(kw["messages"])
        assert "<system>" in msg_blob
        assert "Important rule" in msg_blob
        # the original user message content must still be present
        assert "hello" in msg_blob

    def test_zai_small_system_kept_as_system_field(self):
        """Small system prompt (< 2 KB) stays in the system field for Z.AI."""
        kw = _build_kwargs(
            base_url="https://api.z.ai/api/anthropic",
            system="You are helpful.",  # ~17 chars, well under 2 KB
            messages=[{"role": "user", "content": "hi"}],
        )
        assert kw.get("system") == "You are helpful."

    def test_native_anthropic_large_system_unchanged(self):
        """Native Anthropic must keep the large system in the system field."""
        kw = _build_kwargs(
            base_url="https://api.anthropic.com",
            system=self.BIG_SYSTEM,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert kw.get("system") == self.BIG_SYSTEM
        # must NOT be folded into messages
        assert "<system>" not in str(kw["messages"])

    def test_zai_fold_preserves_content_block_list(self):
        """When the first user message has a content-block list, the system
        block is prepended as a new text block without clobbering the list."""
        kw = _build_kwargs(
            base_url="https://api.z.ai/api/anthropic",
            system=self.BIG_SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "image follows"},
                {"type": "image", "source": {"data": "..."}},
            ]}],
        )
        assert "system" not in kw
        first = kw["messages"][0]
        assert first["role"] == "user"
        content = first["content"]
        assert isinstance(content, list)
        # prepended system block is first
        assert content[0]["type"] == "text"
        assert "<system>" in content[0]["text"]
        # original blocks preserved after it
        assert content[1]["text"] == "image follows"
        assert content[2]["type"] == "image"


# ---------------------------------------------------------------------------
# Regression: native paths untouched
# ---------------------------------------------------------------------------

class TestNativePathsUnaffected:
    def test_native_endpoint_keeps_betas_and_system(self):
        """The whole Z.AI guard must be a no-op for native Anthropic."""
        big_system = "x " * 5000
        kw = _build_kwargs(
            base_url="https://api.anthropic.com",
            system=big_system,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert kw.get("system") == big_system
        assert "<system>" not in str(kw["messages"])

    def test_openai_compat_url_unchanged(self):
        """Z.AI's OpenAI-compat endpoints are not Anthropic wire — must be
        untouched by the anthropic-only fixes."""
        from agent.anthropic_adapter import _is_zai_anthropic_endpoint
        assert not _is_zai_anthropic_endpoint("https://api.z.ai/api/coding/paas/v4")
