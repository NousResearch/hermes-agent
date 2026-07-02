"""Regression test: _resolve_task_provider_model preserves provider when base_url is set.

Before the fix, passing both ``provider`` and ``base_url`` as direct arguments
forced provider to ``"custom"`` regardless of whether the caller specified a
known provider (e.g. ``"alibaba"``).  This caused vision/auxiliary calls to
resolve credentials from the wrong env var (OPENAI_API_KEY instead of
DASHSCOPE_API_KEY) and fail with 401.

The fix preserves the caller's provider when it is a non-empty, non-"auto"
value, so each provider resolves credentials from its own env vars.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pytest


@pytest.fixture
def isolated_home(monkeypatch):
    """Temp HERMES_HOME with clean credential env vars."""
    test_home = tempfile.mkdtemp(prefix="hermes_test_vision_burl_")
    hermes_home = os.path.join(test_home, ".hermes")
    os.makedirs(hermes_home)
    monkeypatch.setenv("HERMES_HOME", hermes_home)

    # Write a minimal config so load_config() doesn't fail.
    with open(os.path.join(hermes_home, "config.yaml"), "w") as fp:
        fp.write("model:\n  default: test-model\n")

    # Strip credential env vars for hermetic tests.
    for k in list(os.environ.keys()):
        if k.endswith("_API_KEY") or k.endswith("_TOKEN"):
            monkeypatch.delenv(k, raising=False)

    yield hermes_home
    shutil.rmtree(test_home, ignore_errors=True)


def _fresh_modules():
    """Drop cached hermes modules so each test reloads cleanly."""
    for mod in list(sys.modules.keys()):
        if mod.startswith(("agent.auxiliary_client", "hermes_cli.config")):
            del sys.modules[mod]


class TestProviderPreservedWithBaseUrl:
    """Direct provider + base_url args must not collapse to 'custom'."""

    def test_known_provider_preserved_with_base_url(self, isolated_home):
        """provider='alibaba' + base_url should return 'alibaba', not 'custom'."""
        _fresh_modules()
        from agent.auxiliary_client import _resolve_task_provider_model

        provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task=None,
            provider="alibaba",
            model="qwen-vl-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-test",
        )
        assert provider == "alibaba", (
            f"Expected 'alibaba' but got '{provider}'. "
            "base_url should not force provider to 'custom' when a known "
            "provider is explicitly specified."
        )
        assert base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert model == "qwen-vl-max"

    def test_auto_provider_still_collapses_to_custom(self, isolated_home):
        """provider='auto' + base_url should still return 'custom' (unchanged behavior)."""
        _fresh_modules()
        from agent.auxiliary_client import _resolve_task_provider_model

        provider, _model, base_url, _key, _mode = _resolve_task_provider_model(
            task=None,
            provider="auto",
            base_url="https://some-proxy.example.com/v1",
        )
        assert provider == "custom", (
            "provider='auto' with base_url should still resolve to 'custom'"
        )

    def test_empty_provider_still_collapses_to_custom(self, isolated_home):
        """provider='' + base_url should still return 'custom' (unchanged behavior)."""
        _fresh_modules()
        from agent.auxiliary_client import _resolve_task_provider_model

        provider, _model, base_url, _key, _mode = _resolve_task_provider_model(
            task=None,
            provider="",
            base_url="https://some-proxy.example.com/v1",
        )
        assert provider == "custom"

    def test_none_provider_still_collapses_to_custom(self, isolated_home):
        """provider=None + base_url should still return 'custom' (unchanged behavior)."""
        _fresh_modules()
        from agent.auxiliary_client import _resolve_task_provider_model

        provider, _model, _base_url, _key, _mode = _resolve_task_provider_model(
            task=None,
            provider=None,
            base_url="https://some-proxy.example.com/v1",
        )
        assert provider == "custom"

    def test_openrouter_provider_preserved_with_base_url(self, isolated_home):
        """Any known provider should be preserved — not just 'alibaba'."""
        _fresh_modules()
        from agent.auxiliary_client import _resolve_task_provider_model

        provider, _model, base_url, _key, _mode = _resolve_task_provider_model(
            task=None,
            provider="openrouter",
            base_url="https://my-proxy.example.com/v1",
            api_key="sk-or-test",
        )
        assert provider == "openrouter"
        assert base_url == "https://my-proxy.example.com/v1"

    def test_base_url_without_provider_still_custom(self, isolated_home):
        """base_url alone (no provider arg) should return 'custom'."""
        _fresh_modules()
        from agent.auxiliary_client import _resolve_task_provider_model

        provider, _model, _base_url, _key, _mode = _resolve_task_provider_model(
            task=None,
            base_url="https://some-proxy.example.com/v1",
        )
        assert provider == "custom"


class TestProviderFromConfigWithBaseUrl:
    """When auxiliary task config specifies provider + base_url,
    the config values must flow through _resolve_task_provider_model
    and preserve the provider (not collapse to 'custom')."""

    def test_config_provider_preserved_via_task(self, isolated_home):
        """Simulate real usage: config.yaml has auxiliary.vision.provider=alibaba
        and auxiliary.vision.base_url, task='vision' triggers config lookup.

        This exercises the config-driven branch (cfg_base_url + cfg_provider,
        no cfg_api_key) rather than the direct-argument branch covered above.
        """
        _fresh_modules()
        # Overwrite the fixture's minimal config with one that mirrors
        # real-world per-task auxiliary usage.
        config_path = os.path.join(os.environ["HERMES_HOME"], "config.yaml")
        with open(config_path, "w") as fp:
            fp.write(
                "model:\n"
                "  default: test-model\n"
                "auxiliary:\n"
                "  vision:\n"
                "    provider: alibaba\n"
                "    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1\n"
            )

        from agent.auxiliary_client import _resolve_task_provider_model

        provider, _model, base_url, api_key, _mode = _resolve_task_provider_model(
            task="vision",
        )
        assert provider == "alibaba", (
            f"Expected config provider 'alibaba' but got '{provider}'. "
            "A config-supplied base_url without api_key must keep the known "
            "provider instead of collapsing to 'custom'."
        )
        assert base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        # No api_key in config → resolution is deferred to the provider's own
        # env vars downstream (see TestCredentialResolutionEndToEnd).
        assert api_key is None


class TestCredentialResolutionEndToEnd:
    """The whole point of preserving provider is to read the right env var.

    _resolve_task_provider_model itself does not read credentials from the
    environment — it only returns the *provider string*.  The credential
    lookup happens downstream in
    ``hermes_cli.auth.resolve_api_key_provider_credentials`` which maps the
    provider to its api_key_env_vars (alibaba → DASHSCOPE_API_KEY).  This test
    chains both steps so the "provider drives the right key" linkage is proven
    end-to-end, not asserted by construction.
    """

    def test_alibaba_provider_reads_dashscope_api_key(self, isolated_home, monkeypatch):
        _fresh_modules()
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-should-not-be-used")

        from agent.auxiliary_client import _resolve_task_provider_model

        # Step 1: resolve the provider (alibaba preserved despite base_url).
        provider, _model, _base_url, api_key, _mode = _resolve_task_provider_model(
            task=None,
            provider="alibaba",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        assert provider == "alibaba"
        # The resolver returns the *passed* api_key (None) — it never touches
        # env vars.  The real credential lookup happens in step 2.
        assert api_key is None

        # Step 2: feed the *resolved* provider (not a hardcoded literal) into
        # the credential resolver so we prove the linkage.
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials(provider)
        assert creds["api_key"] == "sk-dashscope-test", (
            f"Expected DASHSCOPE_API_KEY but got api_key={creds['api_key']!r}. "
            "Provider 'alibaba' should resolve credentials from DASHSCOPE_API_KEY, "
            "not OPENAI_API_KEY."
        )
        assert creds["api_key"] != "sk-openai-should-not-be-used"
        assert creds["source"] == "DASHSCOPE_API_KEY"
