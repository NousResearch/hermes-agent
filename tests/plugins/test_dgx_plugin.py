"""Tests for plugins/dgx/__init__.py — plugin registration and YAML manifest."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Plugin manifest
# ---------------------------------------------------------------------------

class TestPluginYaml:
    def _manifest(self) -> dict:
        path = Path(__file__).parents[2] / "plugins" / "dgx" / "plugin.yaml"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_manifest_is_valid_yaml(self):
        assert self._manifest() is not None

    def test_manifest_has_required_fields(self):
        m = self._manifest()
        for field in ("name", "version", "description"):
            assert field in m, f"missing field: {field}"

    def test_manifest_name_is_dgx(self):
        assert self._manifest()["name"] == "dgx"


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

class TestRegister:
    def _register(self):
        from plugins.dgx import register
        tools = []
        commands = []

        class FakeCtx:
            def register_cli_command(self, **kw):
                commands.append(kw)

            def register_tool(self, name, **kw):
                tools.append(name)

        register(FakeCtx())
        return tools, commands

    def test_registers_dgx_cli_command(self):
        _, commands = self._register()
        names = [c["name"] for c in commands]
        assert "dgx" in names

    def test_dgx_command_has_help_text(self):
        _, commands = self._register()
        dgx_cmd = next(c for c in commands if c["name"] == "dgx")
        assert dgx_cmd.get("help")

    def test_handler_is_callable(self):
        _, commands = self._register()
        dgx_cmd = next(c for c in commands if c["name"] == "dgx")
        assert callable(dgx_cmd["handler_fn"])

    def test_setup_fn_accepts_argparse_subparser(self):
        import argparse
        _, commands = self._register()
        dgx_cmd = next(c for c in commands if c["name"] == "dgx")
        sub = argparse.ArgumentParser()
        dgx_cmd["setup_fn"](sub)  # should not raise


# ---------------------------------------------------------------------------
# dgx-ollama model provider removal (C2)
# ---------------------------------------------------------------------------

class TestDgxOllamaProviderRemoved:
    """C2: the dedicated dgx-ollama provider was dead. It declared
    auth_type="api_key" with env_vars=(), so auth.py's auto-extension skipped
    it (it only adds api_key providers with NON-empty env_vars), and
    resolve_provider() then raised "Unknown provider 'dgx-ollama'". It also
    could not be fixed without either a core edit to auth.py (which the repo's
    own auto-extension comment forbids for new providers) or a new *_BASE_URL
    env var (which would recreate the HERMES_* config violation). The DGX is
    reachable via the working custom path, so the dead provider is removed.
    """

    def test_provider_plugin_source_removed(self):
        # Assert the source module is gone (robust against a stray __pycache__,
        # which Python 3 won't import without the .py anyway).
        provider_dir = Path(__file__).parents[2] / "plugins" / "model-providers" / "dgx-ollama"
        assert not (provider_dir / "__init__.py").exists()
        assert not (provider_dir / "plugin.yaml").exists()

    def test_dgx_endpoint_resolves_to_a_runnable_provider(self):
        # apply_endpoint(ollama) writes model.provider="ollama"; vllm/litellm
        # write "custom". Both resolve to a runnable provider — no dedicated
        # dgx-ollama profile is needed.
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("ollama") == "custom"
        assert resolve_provider("custom") == "custom"
