"""Regression tests for issue #31179.

Before the fix:
  - ``auxiliary.vision.provider: openai`` silently failed to resolve because
    ``openai`` is not a first-class provider in PROVIDER_REGISTRY (only
    ``openai-codex`` for OAuth and ``custom`` for OPENAI_BASE_URL).
  - The vision branch of ``call_llm`` then silently fell back to ``auto``
    which happily picked the user's main provider (e.g. DeepSeek), sending
    image content to a text-only endpoint and producing cryptic
    ``unknown variant 'image_url', expected 'text'`` errors.
  - ``check_vision_requirements`` used the explicit-only path, so
    ``vision_analyze`` disappeared from the tool list while ``browser_vision``
    stayed (its check_fn only validated the browser).

The three fixes covered here:
  1. ``provider: openai`` in auxiliary task config resolves to
     ``custom`` + ``https://api.openai.com/v1``.
  2. The vision auto-detect chain skips the user's main provider when it
     reports ``supports_vision=False`` instead of routing image content to
     a text-only endpoint.
  3. ``check_vision_requirements`` mirrors the runtime fallback chain so
     ``vision_analyze`` shows up whenever the auto chain can serve vision,
     and ``browser_vision`` gates on vision availability as well.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(monkeypatch):
    """Temp HERMES_HOME with config + clean credential env vars."""
    test_home = tempfile.mkdtemp(prefix="hermes_test_31179_")
    hermes_home = os.path.join(test_home, ".hermes")
    os.makedirs(hermes_home)
    monkeypatch.setenv("HERMES_HOME", hermes_home)

    # Strip all credential-shaped env vars so each scenario starts hermetic.
    for k in list(os.environ.keys()):
        if k.endswith("_API_KEY") or k.endswith("_TOKEN"):
            monkeypatch.delenv(k, raising=False)

    yield hermes_home
    shutil.rmtree(test_home, ignore_errors=True)


def _write_config(home: str, text: str) -> None:
    with open(os.path.join(home, "config.yaml"), "w") as fp:
        fp.write(text)


def _fresh_modules():
    """Drop cached hermes modules so each test reloads against current env."""
    for mod in list(sys.modules.keys()):
        if mod.startswith(("agent.auxiliary_client", "agent.image_routing",
                           "tools.vision_tools", "tools.browser_tool",
                           "hermes_cli.config")):
            del sys.modules[mod]


# ---------------------------------------------------------------------------
# Fix 1: provider=openai → custom + api.openai.com/v1
# ---------------------------------------------------------------------------


class TestOpenAiAliasForAuxiliary:
    """``auxiliary.<task>.provider: openai`` should produce a working client."""

    def test_provider_openai_routes_to_openai_dot_com(self, isolated_home, monkeypatch):
        _write_config(isolated_home, """
auxiliary:
  vision:
    provider: openai
    model: gpt-4o-mini
""")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        _fresh_modules()

        from agent.auxiliary_client import _resolve_task_provider_model
        provider, model, base_url, _key, _mode = _resolve_task_provider_model("vision")
        assert provider == "custom"
        assert model == "gpt-4o-mini"
        assert base_url == "https://api.openai.com/v1"

    def test_provider_openai_with_explicit_base_url_preserves_user_endpoint(
        self, isolated_home, monkeypatch
    ):
        """User-supplied base_url wins; alias still normalizes provider name
        to ``custom`` so resolution doesn't hit the unknown-provider path."""
        _write_config(isolated_home, """
auxiliary:
  vision:
    provider: openai
    model: gpt-4o-mini
    base_url: https://my-proxy.example.com/v1
""")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        _fresh_modules()

        from agent.auxiliary_client import _resolve_task_provider_model
        provider, _model, base_url, _key, _mode = _resolve_task_provider_model("vision")
        assert provider == "custom"
        assert base_url == "https://my-proxy.example.com/v1"

    def test_provider_openai_resolves_to_working_client(self, isolated_home, monkeypatch):
        """End-to-end: the resolved client points at api.openai.com."""
        _write_config(isolated_home, """
auxiliary:
  vision:
    provider: openai
    model: gpt-4o-mini
""")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        _fresh_modules()

        from agent.auxiliary_client import resolve_vision_provider_client
        from urllib.parse import urlparse
        provider, client, model = resolve_vision_provider_client()
        assert client is not None, "openai alias should produce a usable client"
        # Exact hostname comparison (not substring) — defends against URLs
        # like ``api.openai.com.evil.example`` and keeps CodeQL happy.
        host = urlparse(str(getattr(client, "base_url", ""))).hostname or ""
        assert host == "api.openai.com", f"expected api.openai.com host, got {host!r}"
        assert model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Fix 2: auto chain skips text-only main providers
# ---------------------------------------------------------------------------


class TestTextOnlyMainSkippedForVision:
    """Vision auto-detect must not return a text-only main-provider client."""

    def test_text_only_main_skipped_when_no_aggregator(self, isolated_home, monkeypatch):
        """DeepSeek main + no aggregator credentials → no client built.

        Pre-fix this silently returned the deepseek client with model
        substitution, producing ``unknown variant 'image_url'`` at call time.
        """
        _write_config(isolated_home, """
model:
  provider: deepseek
  default: deepseek-v4-pro
""")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        _fresh_modules()

        from agent.auxiliary_client import resolve_vision_provider_client
        provider, client, _model = resolve_vision_provider_client(provider="auto")
        assert client is None, (
            f"Vision auto-detect must skip text-only main {provider!r} when "
            "no vision-capable aggregator is available, not return a client "
            "that will fail at API time"
        )

    def test_vision_capable_main_used(self, isolated_home, monkeypatch):
        """Vision-capable main provider should be returned by auto chain."""
        _write_config(isolated_home, """
model:
  provider: anthropic
  default: claude-sonnet-4-6
""")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        _fresh_modules()

        from agent.auxiliary_client import resolve_vision_provider_client
        provider, client, _model = resolve_vision_provider_client(provider="auto")
        assert client is not None
        assert provider == "anthropic"

    def test_exclude_providers_skips_main_in_favor_of_aggregator(self, monkeypatch):
        """A provider in ``exclude_providers`` is skipped so auto-detect lands
        on a vision-capable aggregator instead of the excluded main provider.

        Mirrors the Z.AI Coding-plan case: zai's vision model (glm-5v-turbo)
        is not licensed, so the retry re-resolves with zai excluded and must
        pick an aggregator (OpenRouter) rather than returning zai again.

        The provider-resolution primitives are stubbed so this exercises only
        the ``exclude_providers`` branching — not real credential/pool
        resolution, which carries cross-test cached state and is flaky here.
        """
        _fresh_modules()
        import agent.auxiliary_client as ac

        class _FakeClient:
            def __init__(self, label):
                self.base_url = f"https://{label}.example/v1"

        zai_client = _FakeClient("zai")
        or_client = _FakeClient("openrouter")

        # Force the auto path: main provider = zai, no config/credential reads.
        monkeypatch.setattr(
            ac, "_resolve_task_provider_model",
            lambda *a, **k: ("auto", "", "", "", "chat_completions"))
        monkeypatch.setattr(ac, "_read_main_provider", lambda: "zai")
        monkeypatch.setattr(ac, "_read_main_model", lambda: "glm-5.1")
        # Main-provider vision attempt resolves to a zai client.
        monkeypatch.setattr(
            ac, "resolve_provider_client",
            lambda provider, model=None, **k: (zai_client, model) if provider == "zai" else (None, None))
        # Aggregator chain offers an OpenRouter vision backend.
        monkeypatch.setattr(
            ac, "_resolve_strict_vision_backend",
            lambda candidate, *a, **k: (or_client, "openrouter-vision") if candidate == "openrouter" else (None, None))

        # Without exclusion, auto-detect picks the main provider (zai).
        provider_default, client_default, _ = ac.resolve_vision_provider_client(provider="auto")
        assert client_default is zai_client
        assert provider_default == "zai"

        # Excluding zai must skip it and fall through to the aggregator chain.
        provider_excl, client_excl, _ = ac.resolve_vision_provider_client(
            provider="auto", exclude_providers=frozenset({"zai"})
        )
        assert client_excl is or_client
        assert provider_excl == "openrouter"

    def test_exclude_providers_matches_alias_main_provider(self, monkeypatch):
        """Exclusion sets hold canonical ids; a main provider configured via a
        Z.AI alias ("glm") must still be skipped when "zai" is excluded —
        otherwise the reroute lands straight back on the broken provider."""
        _fresh_modules()
        import agent.auxiliary_client as ac

        class _FakeClient:
            def __init__(self, label):
                self.base_url = f"https://{label}.example/v1"

        or_client = _FakeClient("openrouter")

        monkeypatch.setattr(
            ac, "_resolve_task_provider_model",
            lambda *a, **k: ("auto", "", "", "", "chat_completions"))
        monkeypatch.setattr(ac, "_read_main_provider", lambda: "glm")
        monkeypatch.setattr(ac, "_read_main_model", lambda: "glm-5.1")
        monkeypatch.setattr(
            ac, "_resolve_strict_vision_backend",
            lambda candidate, *a, **k: (or_client, "openrouter-vision") if candidate == "openrouter" else (None, None))

        provider, client, _ = ac.resolve_vision_provider_client(
            provider="auto", exclude_providers=frozenset({"zai"})
        )
        assert client is or_client
        assert provider == "openrouter"

    def test_exclude_providers_does_not_leak_pinned_model_to_aggregator(self, monkeypatch):
        """On a reroute, a config-pinned vision model belongs to the excluded
        provider — the aggregator must get its own known-good default, not a
        zai-only slug it would 404 on."""
        _fresh_modules()
        import agent.auxiliary_client as ac

        class _FakeClient:
            def __init__(self, label):
                self.base_url = f"https://{label}.example/v1"

        or_client = _FakeClient("openrouter")

        # Config pins auxiliary.vision.model to the zai vision model.
        monkeypatch.setattr(
            ac, "_resolve_task_provider_model",
            lambda *a, **k: ("auto", "glm-5v-turbo", "", "", "chat_completions"))
        monkeypatch.setattr(ac, "_read_main_provider", lambda: "zai")
        monkeypatch.setattr(ac, "_read_main_model", lambda: "glm-5.1")
        monkeypatch.setattr(
            ac, "_resolve_strict_vision_backend",
            lambda candidate, *a, **k: (or_client, "openrouter-vision") if candidate == "openrouter" else (None, None))

        provider, client, model = ac.resolve_vision_provider_client(
            provider="auto", exclude_providers=frozenset({"zai"})
        )
        assert provider == "openrouter"
        assert client is or_client
        assert model == "openrouter-vision"

    def test_exclude_providers_does_not_leak_pinned_model_to_main_provider(self, monkeypatch):
        """The pin-vs-default rule must hold on the Step-1 main-provider path
        too, not just the aggregator loop: rerouting with a pinned zai model
        onto a vision-capable non-zai main provider must use that provider's
        own vision default, not the zai-only slug (which it would 404 on)."""
        _fresh_modules()
        import agent.auxiliary_client as ac

        class _FakeClient:
            def __init__(self, label):
                self.base_url = f"https://{label}.example/v1"

        gem_client = _FakeClient("gemini")

        # Config pins auxiliary.vision.model to the zai vision model; the
        # main provider is a different, vision-capable backend.
        monkeypatch.setattr(
            ac, "_resolve_task_provider_model",
            lambda *a, **k: ("auto", "glm-5v-turbo", "", "", "chat_completions"))
        monkeypatch.setattr(ac, "_read_main_provider", lambda: "gemini")
        monkeypatch.setattr(ac, "_read_main_model", lambda: "gemini-3-pro")
        monkeypatch.setattr(
            ac, "_resolve_provider_vision_default",
            lambda provider: "gemini-vision-default")
        monkeypatch.setattr(ac, "_main_model_supports_vision", lambda p, m: True)
        monkeypatch.setattr(
            ac, "resolve_provider_client",
            lambda provider, model=None, **k: (gem_client, model))

        # Under exclusion the main provider serves with its own default...
        provider, client, model = ac.resolve_vision_provider_client(
            provider="auto", exclude_providers=frozenset({"zai"})
        )
        assert provider == "gemini"
        assert client is gem_client
        assert model == "gemini-vision-default"

        # ...while without exclusion the config pin still applies as before.
        provider, _, model = ac.resolve_vision_provider_client(provider="auto")
        assert provider == "gemini"
        assert model == "glm-5v-turbo"

    def test_unknown_capability_does_not_block(self, isolated_home, monkeypatch):
        """When models.dev has no entry, fall back to permissive (attempt the call).

        This keeps new/custom providers working — only providers we have
        cataloged as text-only are skipped.
        """
        _fresh_modules()
        from agent.auxiliary_client import _main_model_supports_vision
        # Bogus provider/model — capability lookup returns None → permissive.
        assert _main_model_supports_vision("nonexistent-provider", "nonexistent-model") is True


# ---------------------------------------------------------------------------
# Fix 3: check_vision_requirements + check_browser_vision_requirements parity
# ---------------------------------------------------------------------------


class TestVisionToolGating:
    """Tool visibility must match runtime capability."""

    def test_check_vision_succeeds_for_aliased_openai(self, isolated_home, monkeypatch):
        """The user's exact reported scenario: provider=openai unhides
        vision_analyze instead of silently dropping it."""
        _write_config(isolated_home, """
auxiliary:
  vision:
    provider: openai
    model: gpt-4o-mini
""")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        _fresh_modules()

        from tools.vision_tools import check_vision_requirements
        assert check_vision_requirements() is True

    def test_check_vision_falls_back_to_auto(self, isolated_home, monkeypatch):
        """Bad explicit provider doesn't hide the tool when auto fallback works.

        Mirrors call_llm's runtime fallback chain.
        """
        _write_config(isolated_home, """
model:
  provider: openrouter
  default: anthropic/claude-sonnet-4
auxiliary:
  vision:
    provider: not-a-real-provider
""")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        _fresh_modules()

        from tools.vision_tools import check_vision_requirements
        assert check_vision_requirements() is True

    def test_check_vision_false_with_text_only_main_and_no_aggregator(
        self, isolated_home, monkeypatch
    ):
        _write_config(isolated_home, """
model:
  provider: deepseek
  default: deepseek-v4-pro
""")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        _fresh_modules()

        from tools.vision_tools import check_vision_requirements
        assert check_vision_requirements() is False

    def test_browser_vision_requires_both_browser_and_vision(self, isolated_home, monkeypatch):
        """``browser_vision`` must not be advertised when vision is unavailable."""
        from unittest.mock import patch

        _write_config(isolated_home, """
model:
  provider: deepseek
  default: deepseek-v4-pro
""")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        _fresh_modules()

        import tools.browser_tool
        # Force the browser side to True so we exercise the vision-gating part.
        with patch.object(tools.browser_tool, "check_browser_requirements", return_value=True):
            assert tools.browser_tool.check_browser_vision_requirements() is False

    def test_browser_vision_false_when_browser_missing(self, isolated_home, monkeypatch):
        from unittest.mock import patch

        _write_config(isolated_home, """
model:
  provider: openrouter
  default: anthropic/claude-sonnet-4
""")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        _fresh_modules()

        import tools.browser_tool
        with patch.object(tools.browser_tool, "check_browser_requirements", return_value=False):
            # Vision available but browser missing → still False.
            assert tools.browser_tool.check_browser_vision_requirements() is False

    def test_browser_vision_true_when_both_available(self, isolated_home, monkeypatch):
        from unittest.mock import patch

        _write_config(isolated_home, """
model:
  provider: openrouter
  default: anthropic/claude-sonnet-4
""")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        _fresh_modules()

        import tools.browser_tool
        with patch.object(tools.browser_tool, "check_browser_requirements", return_value=True):
            assert tools.browser_tool.check_browser_vision_requirements() is True
