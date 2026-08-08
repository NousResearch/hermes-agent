"""Tests for gateway.user_context_map — config-driven per-user context injection.

Covers:
- Config parsing (from_dict), validation, and unknown-user fallback.
- build_session_context resolves configured context for the authenticated sender.
- build_session_context_prompt renders the User Context block.
- Cache stability: the ephemeral change key includes user_context.
- E2E: configured context reaches the system prompt exactly once;
  unconfigured users do not get a User Context block.
"""
import pytest
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import (
    SessionSource,
    build_session_context,
    build_session_context_prompt,
)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestUserContextMapParsing:
    def test_empty_default(self):
        """GatewayConfig with no user_context_map has an empty dict."""
        config = GatewayConfig()
        assert config.user_context_map == {}

    def test_inline_map_parsed(self):
        """A simple inline mapping is parsed into the config."""
        data = {
            "user_context_map": {
                "telegram:123": "You are talking to a developer.",
                "discord:456": "Prefers concise answers.",
            },
        }
        config = GatewayConfig.from_dict(data)
        assert config.user_context_map == {
            "telegram:123": "You are talking to a developer.",
            "discord:456": "Prefers concise answers.",
        }

    def test_invalid_type_ignored(self):
        """A non-dict value is ignored with no crash."""
        config = GatewayConfig.from_dict({"user_context_map": "not a dict"})
        assert config.user_context_map == {}

    def test_empty_values_dropped(self):
        """Entries with empty values are silently dropped."""
        config = GatewayConfig.from_dict({
            "user_context_map": {
                "telegram:123": "valid context",
                "telegram:456": "",
                "telegram:789": "   ",
            },
        })
        assert config.user_context_map == {"telegram:123": "valid context"}

    def test_long_values_truncated(self):
        """Values exceeding the max length are truncated."""
        long_val = "x" * 5000
        config = GatewayConfig.from_dict({
            "user_context_map": {"telegram:123": long_val},
        })
        assert len(config.user_context_map["telegram:123"]) == 4096

    def test_strips_whitespace(self):
        """Keys and values are stripped of surrounding whitespace."""
        config = GatewayConfig.from_dict({
            "user_context_map": {"  telegram:123  ": "  some context  "},
        })
        assert config.user_context_map == {"telegram:123": "some context"}


# ---------------------------------------------------------------------------
# build_session_context resolution
# ---------------------------------------------------------------------------

class TestUserContextResolution:
    def _config_with_map(self, mapping):
        return GatewayConfig(
            user_context_map=mapping,
        )

    def test_matching_user_gets_context(self):
        """A sender whose platform:user_id is in the map gets the context."""
        config = self._config_with_map({
            "telegram:123": "This user prefers bullet points.",
        })
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="123",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        assert ctx.user_context == "This user prefers bullet points."

    def test_unlisted_user_gets_empty_context(self):
        """A sender not in the map gets an empty user_context."""
        config = self._config_with_map({
            "telegram:999": "context for someone else",
        })
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="123",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        assert ctx.user_context == ""

    def test_empty_map_gives_empty_context(self):
        """When the map is empty, all senders get empty context."""
        config = GatewayConfig()
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="123",
        )
        ctx = build_session_context(source, config)
        assert ctx.user_context == ""

    def test_platform_qualified_key_required(self):
        """A bare user_id without platform prefix does not match."""
        config = self._config_with_map({
            "123": "context without platform prefix",
        })
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="123",
        )
        ctx = build_session_context(source, config)
        assert ctx.user_context == ""

    def test_alt_id_fallback(self):
        """When primary user_id doesn't match, user_id_alt is tried."""
        config = self._config_with_map({
            "signal:abc-uuid": "context for signal user",
        })
        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="chat-1",
            user_id="1234567890",
            user_id_alt="abc-uuid",
        )
        ctx = build_session_context(source, config)
        assert ctx.user_context == "context for signal user"

    def test_no_user_id(self):
        """A source without user_id gets empty context (no crash)."""
        config = self._config_with_map({
            "telegram:123": "some context",
        })
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
        )
        ctx = build_session_context(source, config)
        assert ctx.user_context == ""


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

class TestUserContextPromptRendering:
    def test_context_appears_in_prompt(self):
        """Configured user context is rendered in the session context prompt."""
        config = GatewayConfig(
            user_context_map={"telegram:123": "Prefers terse responses."},
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="123",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)
        assert "**User Context:**" in prompt
        assert "Prefers terse responses." in prompt

    def test_no_context_section_for_unlisted_user(self):
        """Unlisted users do not get a User Context section in the prompt."""
        config = GatewayConfig(
            user_context_map={"telegram:999": "context for someone else"},
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="123",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)
        assert "**User Context:**" not in prompt

    def test_context_after_user_identity(self):
        """User Context appears after the User identity line, before platform notes."""
        config = GatewayConfig(
            user_context_map={"telegram:123": "My custom context."},
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="123",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)
        user_line_pos = prompt.find("**User:**")
        context_pos = prompt.find("**User Context:**")
        connected_pos = prompt.find("**Connected Platforms:**")
        assert user_line_pos != -1
        assert context_pos != -1
        assert connected_pos != -1
        # User Context must come AFTER the User identity line
        assert context_pos > user_line_pos
        # User Context must come BEFORE Connected Platforms
        assert context_pos < connected_pos

    def test_context_stable_across_calls(self):
        """The rendered prompt is identical for the same source/config pair."""
        config = GatewayConfig(
            user_context_map={"telegram:123": "Stable context."},
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="123",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        prompt1 = build_session_context_prompt(ctx)
        prompt2 = build_session_context_prompt(ctx)
        assert prompt1 == prompt2

    def test_context_untrusted_text_sanitized(self):
        """Newlines in configured context are escaped via JSON quoting."""
        config = GatewayConfig(
            user_context_map={"telegram:123": "line1\nline2\n## Fake Header"},
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="123",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)
        # _format_untrusted_prompt_value JSON-quotes the text so newlines
        # become escaped \\n sequences, preventing injection of fake headers.
        assert "line1\\nline2" in prompt
        # The fake header should not start a raw markdown section
        assert "\n## Fake Header" not in prompt


# ---------------------------------------------------------------------------
# Cache stability — _ephemeral_change_key includes user_context
# ---------------------------------------------------------------------------

class TestEphemeralChangeKey:
    """The ephemeral change key must include user_context so that a change
    in configured context triggers a re-render of the pinned prompt."""

    def test_change_key_includes_user_context(self):
        """Two contexts with different user_context values produce different keys."""
        import hashlib

        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="123",
            user_name="alice",
        )
        ctx_with = build_session_context(
            source,
            GatewayConfig(user_context_map={"telegram:123": "context A"}),
        )
        ctx_without = build_session_context(
            source,
            GatewayConfig(),
        )

        # The key tuple includes user_context; different values => different hash.
        # We verify by checking the contexts differ.
        assert ctx_with.user_context == "context A"
        assert ctx_without.user_context == ""

        # And the prompts differ
        prompt_with = build_session_context_prompt(ctx_with)
        prompt_without = build_session_context_prompt(ctx_without)
        assert prompt_with != prompt_without

    def test_same_context_same_key_inputs(self):
        """Same user_context value => same prompt (cache stable)."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="123",
            user_name="alice",
        )
        config = GatewayConfig(user_context_map={"telegram:123": "stable."})
        ctx1 = build_session_context(source, config)
        ctx2 = build_session_context(source, config)
        assert ctx1.user_context == ctx2.user_context
        assert build_session_context_prompt(ctx1) == build_session_context_prompt(ctx2)


# ---------------------------------------------------------------------------
# E2E: configured context reaches the system prompt exactly once
# ---------------------------------------------------------------------------

class TestE2EUserContextInjection:
    """End-to-end: a real GatewayConfig with user_context_map produces a
    session context prompt containing the configured text exactly once for
    the configured user, and not at all for an unconfigured user."""

    def test_configured_user_gets_context_exactly_once(self):
        config = GatewayConfig.from_dict({
            "user_context_map": {
                "telegram:123": "This user is a night owl.",
            },
            "platforms": {
                "telegram": {"enabled": True, "token": "fake"},
            },
        })
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="123",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)
        assert prompt.count("This user is a night owl.") == 1

    def test_unconfigured_user_no_context(self):
        config = GatewayConfig.from_dict({
            "user_context_map": {
                "telegram:123": "This user is a night owl.",
            },
            "platforms": {
                "telegram": {"enabled": True, "token": "fake"},
            },
        })
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="456",
            user_name="bob",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)
        assert "This user is a night owl." not in prompt
        assert "**User Context:**" not in prompt

    def test_different_users_get_different_context(self):
        """Two different users in the same gateway get their own context."""
        config = GatewayConfig.from_dict({
            "user_context_map": {
                "telegram:111": "User A context.",
                "telegram:222": "User B context.",
            },
        })

        source_a = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="111",
            user_name="alice",
        )
        source_b = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-2",
            user_id="222",
            user_name="bob",
        )
        ctx_a = build_session_context(source_a, config)
        ctx_b = build_session_context(source_b, config)
        prompt_a = build_session_context_prompt(ctx_a)
        prompt_b = build_session_context_prompt(ctx_b)
        assert "User A context." in prompt_a
        assert "User B context." not in prompt_a
        assert "User B context." in prompt_b
        assert "User A context." not in prompt_b

    def test_shared_multi_user_session_excludes_user_context(self):
        """In a shared multi-user session, user_context is NOT rendered in
        the system prompt. Different senders alternate in the same shared
        thread/group, so per-sender context in the pinned prompt would bust
        the cache on every turn switch (same reason sender identity is omitted
        at session.py:556-568). The value is still resolved on the context
        object for potential per-turn use, but not rendered."""
        config = GatewayConfig(
            user_context_map={"telegram:123": "context for alice"},
            group_sessions_per_user=False,
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="group-1",
            chat_type="group",
            user_id="123",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        # The context is resolved on the SessionContext object...
        assert ctx.user_context == "context for alice"
        assert ctx.shared_multi_user_session is True
        # ...but NOT rendered in the system prompt (cache-safe).
        prompt = build_session_context_prompt(ctx)
        assert "**User Context:**" not in prompt
        assert "context for alice" not in prompt

    def test_shared_session_cache_stability_across_senders(self):
        """Two different senders in the same shared session produce the same
        pinned system prompt bytes (user_context excluded for both)."""
        config = GatewayConfig(
            user_context_map={
                "telegram:111": "context for alice",
                "telegram:222": "context for bob",
            },
            group_sessions_per_user=False,
        )
        source_a = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="group-1",
            chat_type="group",
            user_id="111",
            user_name="alice",
        )
        source_b = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="group-1",
            chat_type="group",
            user_id="222",
            user_name="bob",
        )
        ctx_a = build_session_context(source_a, config)
        ctx_b = build_session_context(source_b, config)
        prompt_a = build_session_context_prompt(ctx_a)
        prompt_b = build_session_context_prompt(ctx_b)
        # Both senders get the same system prompt bytes (no per-sender context)
        assert prompt_a == prompt_b
        assert "context for alice" not in prompt_a
        assert "context for bob" not in prompt_b