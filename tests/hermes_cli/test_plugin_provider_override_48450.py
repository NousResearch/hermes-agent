"""Regression tests for #48450 / review note #48481.

A user plugin may override the ``inference_base_url`` of a hardcoded
provider by setting ``base_url`` on its ``ProviderProfile`` (last-writer-wins,
matching ``register_provider()`` semantics). A plugin that does not set
``base_url`` must leave the hardcoded value untouched.

These tests drive the REAL auto-extend loop in ``hermes_cli/auth.py``. Each
test spawns an isolated subprocess with a temp ``$HERMES_HOME`` containing a
real user provider plugin under ``plugins/model-providers/<name>/__init__.py``,
then imports ``hermes_cli.auth`` fresh so its module-import-time loop runs
against the freshly discovered plugin. They do NOT re-implement the loop body
(as the earlier version did), so they exercise the genuine production path
including the special-provider exclusion (#48481).

The override must be restricted to ordinary api-key providers. Providers with
bespoke token refresh / aggregator resolution (copilot, kimi-coding,
kimi-coding-cn, zai, openrouter, custom) keep their hardcoded endpoint even
if a plugin tries to rewrite it.
"""

from __future__ import annotations

import subprocess
import sys
import json
import os
import textwrap
from pathlib import Path

import pytest

_ARCEE_ORIGINAL = "https://api.arcee.ai/api/v1"
_COPILOT_ORIGINAL = "https://api.githubcopilot.com"


def _write_plugin(plugin_dir: Path, name: str, base_url: str) -> None:
    """Write a user provider plugin that registers ``name`` with ``base_url``."""
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "__init__.py").write_text(
        textwrap.dedent(
            f"""\
            from providers import register_provider, ProviderProfile

            register_provider(ProviderProfile(
                name={name!r},
                display_name={name!r},
                auth_type="api_key",
                env_vars=("DUMMY_{name.upper()}_API_KEY",),
                base_url={base_url!r},
            ))
            """
        )
    )


def _run_with_plugins(home: Path, plugins: dict[str, str]) -> dict[str, str]:
    """Install ``plugins`` ({name: base_url}) under ``$HERMES_HOME`` and run a
    fresh ``hermes_cli.auth`` import in an isolated subprocess. Returns the
    ``inference_base_url`` of every requested provider name.
    """
    mp = home / "plugins" / "model-providers"
    for name, base_url in plugins.items():
        _write_plugin(mp / name, name, base_url)

    # Probe script: fresh import of hermes_cli.auth against this HERMES_HOME.
    probe = home / "_probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import os, sys, json
            names = json.loads(os.environ["_PROBE_NAMES"])
            import hermes_cli.auth
            from hermes_cli.auth import PROVIDER_REGISTRY
            out = {n: PROVIDER_REGISTRY[n].inference_base_url for n in names if n in PROVIDER_REGISTRY}
            print(json.dumps(out))
            """
        )
    )
    repo_root = Path(__file__).resolve().parent.parent.parent
    env = {
        **os.environ,
        "HERMES_HOME": str(home),
        "_PROBE_NAMES": json.dumps(sorted(plugins)),
        "PYTHONPATH": str(repo_root),
    }
    result = subprocess.run(
        [sys.executable, str(probe)],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    if result.returncode != 0:
        pytest.fail(
            f"probe subprocess failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestPluginOverridesHardcodedInferenceBaseUrl:
    """#48450 core contract: a user plugin with the same name as a hardcoded
    provider can override the runtime ``inference_base_url`` by setting
    ``base_url`` on its profile.
    """

    def test_plugin_base_url_overrides_arcee(self, tmp_path):
        """A user plugin targeting the hardcoded ``arcee`` provider points the
        runtime ``inference_base_url`` at a self-hosted proxy without editing
        ``hermes_cli/auth.py``."""
        out = _run_with_plugins(tmp_path / "home", {"arcee": "https://proxy.example.com/arcee/v1"})
        assert out["arcee"] == "https://proxy.example.com/arcee/v1"

    def test_plugin_without_base_url_leaves_hardcoded_alone(self, tmp_path):
        """A plugin registering the same name without ``base_url`` must not
        modify the hardcoded ``inference_base_url``."""
        out = _run_with_plugins(tmp_path / "home", {"arcee": ""})
        assert out["arcee"] == _ARCEE_ORIGINAL

    def test_plugin_does_not_touch_other_providers(self, tmp_path):
        """The override is scoped to the plugin's own name — unrelated
        hardcoded providers keep their original ``inference_base_url``."""
        out = _run_with_plugins(tmp_path / "home", {"arcee": "https://proxy.example.com/arcee/v1"})
        assert out["arcee"] == "https://proxy.example.com/arcee/v1"
        # Sibling hardcoded providers are unaffected (still present, non-empty).
        for sibling in ("xai", "anthropic", "deepseek", "google", "openai"):
            if sibling in out:
                assert out[sibling], f"{sibling} should keep its non-empty base URL"

    def test_special_providers_not_overridden_by_plugin(self, tmp_path):
        """#48481: a plugin named like a bespoke-resolution provider (copilot,
        kimi-coding, kimi-coding-cn, zai) must NOT rewrite that provider's
        hardcoded endpoint, even when it sets ``base_url``."""
        plugins = {
            "copilot": "https://evil.example.com/v1",
            "kimi-coding": "https://evil.example.com/v1",
            "kimi-coding-cn": "https://evil.example.com/v1",
            "zai": "https://evil.example.com/v1",
            "arcee": "https://proxy.example.com/arcee/v1",  # control: should be overridden
        }
        out = _run_with_plugins(tmp_path / "home", plugins)
        assert out["copilot"] == _COPILOT_ORIGINAL
        assert out["kimi-coding"] != "https://evil.example.com/v1"
        assert out["kimi-coding-cn"] != "https://evil.example.com/v1"
        assert out["zai"] != "https://evil.example.com/v1"
        # Control still works.
        assert out["arcee"] == "https://proxy.example.com/arcee/v1"

    def test_plugin_for_new_provider_still_creates_entry(self, tmp_path):
        """A plugin whose name is NOT in ``PROVIDER_REGISTRY`` still gets a
        brand-new entry (existing path, not regressed)."""
        new_name = "totally-new-provider-48450"
        new_url = "https://new.example.com/v1"
        out = _run_with_plugins(tmp_path / "home", {new_name: new_url})
        assert out[new_name] == new_url
