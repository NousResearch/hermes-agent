"""Regression tests: a ``-m provider/model`` value whose provider prefix does
NOT resolve for the running profile must fail loudly at startup rather than
silently stripping the prefix and running the bare model on the default
provider.

Bug (2026-07-16): a kanban worker spawned with ``-m claude-apr/claude-fable-5``
on a profile WITHOUT the model-providers plugins symlink resolved to
``Provider: openai-codex, Model: claude-fable-5`` — the ``claude-apr/`` prefix
(a provider name, not a vendor namespace) was stripped and the bare model sent
to the profile's default provider (openai-codex), producing a confusing
HTTP 400 "model not supported with ChatGPT account" x6 crash loop. The failure
was invisible until reading worker logs.

The fix (in ``cli.py::_normalize_model_for_provider``) distinguishes:
  - a benign VENDOR namespace (``anthropic/…``, ``openai/…``, ``meta-llama/…``)
    → strip to the bare model as before, and
  - a PROVIDER-qualified model whose provider does not resolve here
    (``claude-apr/…``) → raise a loud ValueError naming the provider and the
    plugin/credential_pool hint.
"""
from unittest.mock import patch

import pytest


def _make_cli(model, provider="openai-codex"):
    """Create a HermesCLI with a clean config whose default provider is
    openai-codex (the incident's default), minimal tool mocking."""
    import cli as _cli_mod
    from cli import HermesCLI

    _clean_config = {
        "model": {
            "default": "gpt-5.3-codex",
            "base_url": "",
            "provider": provider,
        },
        "display": {"compact": False, "tool_progress": "all", "resume_display": "full"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    with (
        patch("cli.get_tool_definitions", return_value=[]),
        patch.dict("os.environ", clean_env, clear=False),
        patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}),
    ):
        cli = HermesCLI(model=model, compact=True, max_turns=1)
    return cli


class TestUnknownProviderPrefixFailsLoud:
    def test_unresolvable_provider_prefix_raises(self):
        """``claude-apr/claude-fable-5`` on openai-codex, no plugin → raise."""
        from hermes_cli.auth import AuthError

        cli = _make_cli("claude-apr/claude-fable-5")
        # Force the provider resolver to report the plugin is absent — exactly
        # the profile-without-symlink state the incident happened on.
        with patch(
            "hermes_cli.auth.resolve_provider",
            side_effect=AuthError("Unknown provider 'claude-apr'.", code="invalid_provider"),
        ):
            with pytest.raises(ValueError) as exc:
                cli._normalize_model_for_provider("openai-codex")
        msg = str(exc.value)
        # The message must name the offending provider, refuse the substitution,
        # and hint at the plugin / credential_pool remedy.
        assert "claude-apr" in msg
        assert "does not resolve" in msg
        assert "plugins/model-providers" in msg or "credential_pool" in msg
        # The model must be left UNCHANGED (not stripped onto codex).
        assert cli.model == "claude-apr/claude-fable-5"

    def test_error_names_the_default_provider_it_refused_to_use(self):
        from hermes_cli.auth import AuthError

        cli = _make_cli("claude-apr/claude-fable-5")
        with patch(
            "hermes_cli.auth.resolve_provider",
            side_effect=AuthError("Unknown provider 'claude-apr'.", code="invalid_provider"),
        ):
            with pytest.raises(ValueError) as exc:
                cli._normalize_model_for_provider("openai-codex")
        assert "openai-codex" in str(exc.value)


class TestResolvableProviderPrefixStillStrips:
    def test_provider_prefix_that_resolves_is_stripped(self):
        """When the ``claude-apr`` plugin IS present (resolve_provider
        succeeds), the prefix strips to the bare model — no raise."""
        cli = _make_cli("claude-apr/claude-fable-5")
        with patch("hermes_cli.auth.resolve_provider", return_value="claude-apr"):
            changed = cli._normalize_model_for_provider("openai-codex")
        assert changed is True
        assert cli.model == "claude-fable-5"


class TestVendorPrefixesStillStrip:
    """Benign vendor/model slugs must keep their historical strip behaviour —
    the guard must not regress these."""

    @pytest.mark.parametrize("model,expected", [
        ("openai/gpt-5.4", "gpt-5.4"),
        ("anthropic/claude-opus-4.6", "claude-opus-4.6"),
        ("meta-llama/llama-4-scout", "llama-4-scout"),
    ])
    def test_vendor_prefix_stripped_without_raise(self, model, expected):
        cli = _make_cli(model)
        changed = cli._normalize_model_for_provider("openai-codex")
        assert changed is True
        assert cli.model == expected

    def test_deepseek_provider_prefix_resolves_and_strips(self):
        """``deepseek`` is a known vendor namespace (both a key and a value in
        ``_VENDOR_PREFIXES``), so the guard short-circuits on the vendor
        fast-path and strips WITHOUT ever calling ``resolve_provider`` — this
        exercises the vendor-namespace fast-path, not the resolver path."""
        cli = _make_cli("deepseek/deepseek-v4-flash")
        changed = cli._normalize_model_for_provider("openai-codex")
        assert changed is True
        assert cli.model == "deepseek-v4-flash"


class TestIsKnownVendorNamespace:
    """Unit coverage for the vendor-vs-provider discriminator helper."""

    @pytest.mark.parametrize("prefix", [
        "anthropic", "openai", "google", "deepseek", "meta-llama",
        "x-ai", "z-ai", "qwen", "xiaomi",              # canonical vendor slugs
        "claude", "gpt", "gemini", "llama", "grok", "glm",  # first-token aliases
        "ANTHROPIC", " OpenAI ",                        # case / whitespace tolerant
    ])
    def test_recognized_vendor_namespaces(self, prefix):
        from hermes_cli.model_normalize import is_known_vendor_namespace

        assert is_known_vendor_namespace(prefix) is True

    @pytest.mark.parametrize("prefix", [
        "claude-apr",      # api-proxy pool relay PLUGIN — a provider, not a vendor
        "claude-apx-0",
        "claude-bpr",
        "openai-codex",    # a provider id, not a vendor namespace
        "zai",             # provider id (the vendor slug is z-ai)
        "opencode-go",
        "my-custom-provider",
        "",
        "   ",
    ])
    def test_non_vendor_prefixes_rejected(self, prefix):
        from hermes_cli.model_normalize import is_known_vendor_namespace

        assert is_known_vendor_namespace(prefix) is False
