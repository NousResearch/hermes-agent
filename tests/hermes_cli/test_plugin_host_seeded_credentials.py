"""plugins.isolation: host — a hosted provider that seeds its own pool rows gets plain data and the
ambient token it declares, exactly as it would in-process."""

import os
import sys

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod

SEEDING_PLUGIN = '''
import os
from providers import register_provider
from providers.base import ProviderProfile

class Seeds(ProviderProfile):
    def seed_pool_credentials(self, *, suppressed_sources, explicitly_configured):
        token = os.environ.get("GH_TOKEN", "")
        if not token or "env:GH_TOKEN" in suppressed_sources:
            return []
        return [{"source": "env:GH_TOKEN", "access_token": f"minted-{token}-{explicitly_configured}",
                 "base_url": None, "label": f"pid {os.getpid()}"}]

register_provider(Seeds(name="seedhost", base_url="https://seedhost.example/v1",
                        env_vars=("GH_TOKEN",), ambient_credentials=True))
'''


def _install(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "model-providers" / "seedhost"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: seedhost\nkind: model-provider\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(SEEDING_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"isolation": "host"}}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GH_TOKEN", "shell-only")  # exported in the shell, absent from .env


def _shutdown():
    host = getattr(plugins_mod.get_plugin_manager(), "_plugin_host_instance", None)
    if host is not None:
        host.shutdown()
    for name in [m for m in sys.modules if m.endswith("_seedhost")]:
        sys.modules.pop(name, None)


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
def test_hosted_profile_seeds_the_pool_from_a_shell_exported_ambient_token(tmp_path, monkeypatch):
    import providers
    from agent.credential_pool import load_pool

    _install(tmp_path, monkeypatch)
    try:
        profile = providers.get_provider_profile("seedhost")
        assert type(profile).__module__ == "hermes_cli.plugin_host_profiles"
        rows = [(e.source, e.access_token, e.label) for e in load_pool("seedhost").entries()]
        assert rows == [("env:GH_TOKEN", "minted-shell-only-True", rows[0][2])]
        assert rows[0][2] != f"pid {os.getpid()}"  # seeded by the plugin host, not in-process
    finally:
        _shutdown()


@pytest.mark.platforms("any")
def test_hosted_profile_skips_a_suppressed_source(tmp_path, monkeypatch):
    from agent.credential_pool import load_pool
    from hermes_cli.auth import suppress_credential_source

    _install(tmp_path, monkeypatch)
    suppress_credential_source("seedhost", "env:GH_TOKEN")
    try:
        assert load_pool("seedhost").entries() == []
    finally:
        _shutdown()


def test_a_routed_home_host_never_inherits_the_launch_shell_token(tmp_path, monkeypatch):
    import providers
    from hermes_cli.plugin_host import _hosted_ambient_credentials
    from hermes_cli.plugin_host_profiles import _profile_proxy
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from providers.base import ProviderProfile

    launch, coder, writer = tmp_path / "launch", tmp_path / "coder", tmp_path / "writer"
    for home in (launch, coder, writer):
        home.mkdir()
    (writer / ".env").write_text("GH_TOKEN=writer-own\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("GH_TOKEN", "launch-shell")
    hosted = _profile_proxy(ProviderProfile, str(tmp_path), "m", {
        "name": "amb", "fields": {"env_vars": ["GH_TOKEN"], "ambient_credentials": True}})
    monkeypatch.setattr(providers, "list_providers", lambda: [hosted, ProviderProfile(
        name="inproc", env_vars=("GITHUB_TOKEN",), ambient_credentials=True)])
    monkeypatch.setenv("GITHUB_TOKEN", "never-for-an-in-process-profile")

    def env_for(home):
        token = set_hermes_home_override(home)
        try:
            return _hosted_ambient_credentials(home)
        finally:
            reset_hermes_home_override(token)

    assert env_for(launch) == {"GH_TOKEN": "launch-shell"}
    assert env_for(coder) == {}
    assert env_for(writer) == {"GH_TOKEN": "writer-own"}
