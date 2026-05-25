"""Tests for agent/credential_sources.py — RemovalStep, RemovalResult, registry."""

from agent.credential_sources import (
    RemovalResult,
    RemovalStep,
    register,
    find_removal_step,
)


# ── RemovalResult ─────────────────────────────────────────────────────────────

class TestRemovalResult:
    """RemovalResult dataclass — default values and construction."""

    def test_defaults(self):
        """Default RemovalResult has empty lists and suppress=True."""
        rr = RemovalResult()
        assert rr.cleaned == []
        assert rr.hints == []
        assert rr.suppress is True

    def test_custom_values(self):
        """All fields can be set explicitly."""
        rr = RemovalResult(
            cleaned=["Cleared X from .env"],
            hints=["Note: check shell"],
            suppress=False,
        )
        assert rr.cleaned == ["Cleared X from .env"]
        assert rr.hints == ["Note: check shell"]
        assert rr.suppress is False

    def test_partial_construction(self):
        """Fields not provided use defaults."""
        rr = RemovalResult(cleaned=["done"])
        assert rr.cleaned == ["done"]
        assert rr.hints == []
        assert rr.suppress is True


# ── RemovalStep ───────────────────────────────────────────────────────────────

class TestRemovalStep:
    """RemovalStep dataclass and its matches() predicate."""

    def _dummy_remove(self, provider, removed):
        return RemovalResult()

    def test_matches_exact_provider_and_source(self):
        """Literal match on provider and source_id."""
        step = RemovalStep(
            provider="xai",
            source_id="env:XAI_API_KEY",
            remove_fn=self._dummy_remove,
        )
        assert step.matches("xai", "env:XAI_API_KEY") is True

    def test_matches_wrong_provider(self):
        """Mismatched provider returns False."""
        step = RemovalStep(
            provider="xai",
            source_id="env:XAI_API_KEY",
            remove_fn=self._dummy_remove,
        )
        assert step.matches("openai", "env:XAI_API_KEY") is False

    def test_matches_wrong_source(self):
        """Mismatched source_id returns False."""
        step = RemovalStep(
            provider="xai",
            source_id="env:XAI_API_KEY",
            remove_fn=self._dummy_remove,
        )
        assert step.matches("xai", "device_code") is False

    def test_matches_wildcard_provider(self):
        """provider='*' matches any provider."""
        step = RemovalStep(
            provider="*",
            source_id="manual",
            remove_fn=self._dummy_remove,
        )
        assert step.matches("anthropic", "manual") is True
        assert step.matches("xai", "manual") is True
        assert step.matches("anything", "manual") is True

    def test_matches_wildcard_wrong_source(self):
        """provider='*' still requires source_id match."""
        step = RemovalStep(
            provider="*",
            source_id="manual",
            remove_fn=self._dummy_remove,
        )
        assert step.matches("xai", "env:KEY") is False

    def test_matches_with_match_fn(self):
        """match_fn overrides literal source_id comparison."""
        step = RemovalStep(
            provider="xai",
            source_id="unused",  # ignored when match_fn is set
            remove_fn=self._dummy_remove,
            match_fn=lambda src: src.startswith("env:"),
        )
        assert step.matches("xai", "env:XAI_API_KEY") is True
        assert step.matches("xai", "env:OPENAI_KEY") is True
        assert step.matches("xai", "device_code") is False

    def test_matches_with_match_fn_and_wrong_provider(self):
        """match_fn doesn't bypass provider check."""
        step = RemovalStep(
            provider="xai",
            source_id="env",
            remove_fn=self._dummy_remove,
            match_fn=lambda src: src.startswith("env:"),
        )
        assert step.matches("openai", "env:XAI_API_KEY") is False

    def test_matches_match_fn_with_wildcard_provider(self):
        """Wildcard provider + match_fn matches any provider with matching source."""
        step = RemovalStep(
            provider="*",
            source_id="",
            remove_fn=self._dummy_remove,
            match_fn=lambda src: "config" in src,
        )
        assert step.matches("any", "config:my-custom") is True
        assert step.matches("other", "env:KEY") is False

    def test_description_field(self):
        """description is an optional metadata field."""
        step = RemovalStep(
            provider="test",
            source_id="src",
            remove_fn=self._dummy_remove,
            description="Test removal step",
        )
        assert step.description == "Test removal step"

    def test_description_defaults_to_empty(self):
        """description defaults to empty string."""
        step = RemovalStep(
            provider="test",
            source_id="src",
            remove_fn=self._dummy_remove,
        )
        assert step.description == ""


# ── Registry (register + find_removal_step) ────────────────────────────────────

class TestCredentialSourceRegistry:
    """register() and find_removal_step() — first-match-wins semantics."""

    def _dummy_remove(self, provider, removed):
        return RemovalResult()

    def test_find_returns_none_for_unregistered(self):
        """Unregistered (provider, source) returns None."""
        # Use a truly unmatchable source (env:* catch-all is auto-registered)
        assert find_removal_step("nonexistent", "completely:UNMATCHED-source") is None

    def test_register_and_find(self):
        """Registered steps are findable."""
        step = RemovalStep(
            provider="test-prov",
            source_id="test-src",
            remove_fn=self._dummy_remove,
        )
        register(step)
        found = find_removal_step("test-prov", "test-src")
        assert found is step

    def test_first_match_wins(self):
        """When multiple steps match, the first registered wins."""
        step1 = RemovalStep(
            provider="first-prov",
            source_id="first-src",
            remove_fn=self._dummy_remove,
            description="first",
        )
        step2 = RemovalStep(
            provider="first-prov",
            source_id="first-src",
            remove_fn=self._dummy_remove,
            description="second",
        )
        register(step1)
        register(step2)
        found = find_removal_step("first-prov", "first-src")
        assert found.description == "first"

    def test_match_fn_registered_step(self):
        """Steps with match_fn work through find_removal_step."""
        step = RemovalStep(
            provider="*",
            source_id="",
            remove_fn=self._dummy_remove,
            match_fn=lambda src: src.startswith("custom-prefix:"),
            description="custom prefix matcher",
        )
        register(step)
        found = find_removal_step("any-provider", "custom-prefix:my-key")
        assert found is step
        # Non-matching source — but env:* catch-all is auto-registered,
        # so test with a source that matches neither
        assert find_removal_step("any-provider", "zzz:no-match") is None
