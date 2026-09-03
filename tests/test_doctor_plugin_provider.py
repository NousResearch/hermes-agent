"""Test that hermes doctor accepts runtime model-provider plugins.

Regression for #68164: Doctor must not reject providers resolved by the
runtime plugin registry. This test creates a temporary plugin, configures
it, runs Doctor, and verifies no unknown-provider diagnostic appears.
"""

import contextlib
import io
import sys
import types
from argparse import Namespace


def _clear_provider_caches():
    """Force providers/__init__.py to re-discover on next list_providers()."""
    import providers as _pkg

    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._PROVIDER_LIST_CACHE = None
    _pkg._discovered = False
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("plugins.model_providers")
            or mod.startswith("_hermes_user_provider")
        ):
            del sys.modules[mod]


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
    supports_health_check=False,
)

register_provider(test_provider)
""",
        encoding="utf-8",
    )
    (plugin_dir / "plugin.yaml").write_text(
        "name: test-plugin\n"
        "kind: model-provider\n"
        "version: 0.0.1\n"
        "description: Runtime plugin used by Doctor validation regression\n",
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

    # Set up environment for Doctor. run_doctor reads the module-level
    # HERMES_HOME constant; list_providers() reads get_hermes_home() / env.
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(hermes_home))
    monkeypatch.setattr(doctor_mod, "_APIKEY_PROVIDERS_CACHE", None)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (tmp_path / "project").mkdir(exist_ok=True)

    # Re-scan after HERMES_HOME points at the temp plugin.
    _clear_provider_caches()

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
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
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
    # Current Doctor wording (must stay in sync with run_doctor()):
    #   "model.provider 'test-plugin' is not a recognised provider"
    #   "model.provider 'test-plugin' is unknown."
    assert "is not a recognised provider" not in output, (
        f"Doctor rejected runtime plugin provider. Expected test-plugin to be "
        f"recognized by the plugin registry, but got unknown-provider diagnostic.\n"
        f"Output:\n{output}"
    )
    assert "model.provider 'test-plugin' is unknown" not in output, (
        f"Doctor rejected runtime plugin provider. Expected test-plugin to be "
        f"recognized by the plugin registry, but got unknown-provider diagnostic.\n"
        f"Output:\n{output}"
    )

    # Sanity check: verify Doctor actually ran the provider section
    assert "model.provider" in output or "Provider" in output, (
        f"Doctor output missing provider section. Test may not be exercising "
        f"the right code path.\nOutput:\n{output}"
    )
