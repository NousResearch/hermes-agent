"""Tests for the ``model.preserve_dots`` opt-in override.

Third-party Anthropic-compatible proxies (OneAPI, LiteLLM, FastGPT,
company-internal gateways, etc.) often register model IDs that contain
dots — e.g. ``Claude Opus 4.6`` routed to AWS Bedrock. Without an opt-in
override the default ``_anthropic_preserve_dots`` heuristic returns
False for any URL Hermes doesn't specifically recognize, and
``normalize_model_name`` converts dots to hyphens, producing
``Claude Opus 4-6`` — which the proxy cannot match to a registered
channel (HTTP 503 "no available channel").

These tests lock in the behaviour of the ``model.preserve_dots: true``
config opt-in:

- ``True``  → always preserve dots, regardless of provider/base_url.
- ``False`` → always mangle dots, overriding even the built-in
  allowlist (escape hatch in case the allowlist is wrong for a user).
- Missing / non-bool → fall back to the existing allowlist (no
  behaviour change for users who don't set the flag).
"""

from types import SimpleNamespace


class TestPreserveDotsOverrideTrue:
    """``model.preserve_dots: true`` must force dot-preservation even when
    neither the provider nor the base URL matches the built-in allowlist."""

    def test_custom_proxy_with_override_preserves_dots(self):
        """The motivating case: a generic third-party proxy whose URL
        Hermes doesn't recognize. Without the override, dots would be
        mangled. With it, they are preserved."""
        agent = SimpleNamespace(
            provider="custom",
            base_url="https://proxy.example.com",
            _model_preserve_dots_override=True,
        )
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_override_true_wins_over_anthropic_native(self):
        """Escape hatch: if the user explicitly sets ``preserve_dots:
        true`` while pointing at a URL Hermes would normally mangle
        (even api.anthropic.com), the explicit choice wins."""
        agent = SimpleNamespace(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            _model_preserve_dots_override=True,
        )
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True


class TestPreserveDotsOverrideFalse:
    """``model.preserve_dots: false`` must force dot-mangling even when
    the provider/base_url allowlist would otherwise preserve dots —
    this is an escape hatch for cases where the allowlist is wrong."""

    def test_override_false_wins_over_bedrock_provider(self):
        """Even with ``provider="bedrock"`` (which normally preserves
        dots), an explicit False must override."""
        agent = SimpleNamespace(
            provider="bedrock",
            base_url="",
            _model_preserve_dots_override=False,
        )
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is False

    def test_override_false_wins_over_dashscope_url(self):
        """Base-URL-based allowlist must also be overridden by an
        explicit ``preserve_dots: false``."""
        agent = SimpleNamespace(
            provider="custom",
            base_url="https://dashscope.aliyuncs.com",
            _model_preserve_dots_override=False,
        )
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is False


class TestPreserveDotsOverrideUnset:
    """Missing or non-bool values must not change behaviour — the
    allowlist is consulted exactly as before this feature landed."""

    def test_no_attribute_falls_back_to_allowlist_true(self):
        """A ``SimpleNamespace`` without the override attribute exercises
        the ``getattr(..., None)`` default path. Provider is on the
        allowlist → dots preserved."""
        agent = SimpleNamespace(provider="bedrock", base_url="")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_no_attribute_falls_back_to_allowlist_false(self):
        """Same as above but for a URL not on the allowlist — dots are
        still mangled, preserving pre-feature behaviour for users who
        don't opt in."""
        agent = SimpleNamespace(provider="anthropic", base_url="https://api.anthropic.com")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is False

    def test_none_override_falls_back_to_allowlist(self):
        """Explicit ``None`` (the value written by ``__init__`` when
        ``preserve_dots`` is absent or non-bool) must behave the same
        as a missing attribute."""
        agent = SimpleNamespace(
            provider="bedrock",
            base_url="",
            _model_preserve_dots_override=None,
        )
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_non_bool_override_is_ignored(self):
        """Only ``bool`` values are honored. A stray string from
        hand-edited YAML (e.g. ``preserve_dots: "yes"``) must not
        activate the override — the ``__init__`` normalization should
        have written ``None`` instead, but defend in depth."""
        agent = SimpleNamespace(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            _model_preserve_dots_override="yes",  # type: ignore[arg-type]
        )
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is False
