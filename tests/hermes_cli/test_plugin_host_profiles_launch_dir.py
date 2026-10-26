"""Plugin-host profile extraction runs the Hermes tree's code, never a same-named package in the cwd."""

import json

import pytest

from hermes_cli.plugin_host_profiles import load_hosted_profiles

PROVIDER_PLUGIN = '''
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(name="cwdprobe", base_url="https://cwdprobe.example/v1"))
'''

# A launch directory (e.g. a cloned repository) that happens to contain a ``hermes_cli`` package.
SHADOW_INIT = '''
from pathlib import Path
Path(__file__).resolve().parents[1].joinpath("shadow-imported.txt").write_text("imported")
'''
SHADOW_CHILD = 'import json\nprint(json.dumps({"profiles": []}))\n'


@pytest.mark.platforms("any")  # a real child interpreter: sys.path rules are per-OS
def test_profile_extraction_ignores_a_hermes_cli_package_in_the_launch_directory(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    plugin_dir = home / "plugins" / "model-providers" / "cwdprobe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(PROVIDER_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(json.dumps({"plugins": {"isolation": "host"}}), encoding="utf-8")
    launch_dir = tmp_path / "cloned-repo"
    (launch_dir / "hermes_cli").mkdir(parents=True)
    (launch_dir / "hermes_cli" / "__init__.py").write_text(SHADOW_INIT, encoding="utf-8")
    (launch_dir / "hermes_cli" / "plugin_host_child.py").write_text(SHADOW_CHILD, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(launch_dir)

    profiles = load_hosted_profiles(plugin_dir, "_hermes_test_cwdprobe")

    assert not (launch_dir / "shadow-imported.txt").exists()
    assert [p.name for p in profiles] == ["cwdprobe"]
    assert profiles[0].base_url == "https://cwdprobe.example/v1"
