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
                           "agent.models_dev",
                           "tools.vision_tools", "tools.browser_tool",
                           "hermes_cli.config", "hermes_cli.plugins")):
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
