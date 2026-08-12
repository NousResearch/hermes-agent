"""Test that hermes doctor accepts runtime model-provider plugins.

Regression for #68164: Doctor must not reject providers resolved by the
runtime plugin registry. This test creates a temporary plugin, configures
it, runs Doctor, and verifies no unknown-provider diagnostic appears.
"""

import io
import sys
import types
import contextlib
from argparse import Namespace
from pathlib import Path


def test_doctor_accepts_runtime_model_provider_plugin(monkeypatch, tmp_path):
    """Doctor must not reject providers resolved by the runtime plugin registry."""
    from hermes_cli import doctor as doctor_mod

    # 1. Create temp HERMES_HOME
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)

    # 2. Create a model-provider plugin
    plugin_dir = hermes_home / "plugins" / "model-providers" / "test-plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)

    # Write plugin __init__.py that registers a provider profile
    (plugin_dir / "__init__.py").write_text(
        """
from providers import register_provider
from providers.base import ProviderProfile

test_provider = ProviderProfile(
    name="test-plugin",
    aliases=("test",),
    api_mode="chat_completions",
    env_vars=("TEST_PLUGIN_API_KEY",),
    base_url="https://api.test-plugin.example/v1",
    auth_type="api_key",
    supports_tools=True,
)

register_provider(test_provider)
""",
        encoding="utf-8",
    )

    # 3. Configure it as model.provider in config.yaml
    (hermes_home / "config.yaml").write_text(
        """model:
  provider: test-plugin
  default: test-model
memory: {}
""",
        encoding="utf-8",
    )

    # Set up environment for Doctor
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(hermes_home))
    (tmp_path / "project").mkdir(exist_ok=True)

    # Stub tool availability checks
    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    # Stub auth checks to avoid real API calls
    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    # 4. Run Doctor and capture output
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    output = buf.getvalue()

    # 5. Assert the unknown-provider diagnostic is absent
    # If the plugin wasn't recognized, Doctor would show:
    # "Unknown provider: test-plugin"
    assert "Unknown provider" not in output, (
        f"Doctor rejected runtime plugin provider. Expected test-plugin to be "
        f"recognized by the plugin registry, but got unknown-provider diagnostic.\n"
        f"Output:\n{output}"
    )

    # Sanity check: verify Doctor actually ran the provider section
    assert "Model Provider" in output or "Provider" in output, (
        f"Doctor output missing provider section. Test may not be exercising "
        f"the right code path.\nOutput:\n{output}"
    )
