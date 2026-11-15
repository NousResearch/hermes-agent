"""plugins.isolation: host — a profile that cannot cross the host boundary never takes its siblings down."""

import logging
import os
import sys

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod

TWO_PROFILE_PLUGIN = '''
import os
from providers import register_provider
from providers.base import ProviderProfile

class Hosted(ProviderProfile):
    def build_extra_body(self, *, session_id=None, **context):
        return {"pid": os.getpid()}

class Local(ProviderProfile):
    def create_client(self, **kwargs):
        return {"pid": os.getpid()}

register_provider(Hosted(name="mixhosted", base_url="https://mixhosted.example/v1"))
register_provider(Local(name="mixlocal", base_url="acp://mixlocal"))
'''


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
def test_create_client_profile_loads_in_process_and_its_sibling_stays_hosted(tmp_path, monkeypatch, caplog):
    import providers

    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "model-providers" / "mixpair"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: mixpair\nkind: model-provider\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(TWO_PROFILE_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"isolation": "host"}}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))

    with caplog.at_level(logging.WARNING, logger="providers"):
        hosted = providers.get_provider_profile("mixhosted")
        local = providers.get_provider_profile("mixlocal")
    try:
        assert hosted is not None and local is not None
        assert local.create_client()["pid"] == os.getpid()
        assert hosted.build_extra_body(session_id="s")["pid"] != os.getpid()
        # The in-process import registered ONLY the profile that needs it.
        assert type(hosted).__module__ == "hermes_cli.plugin_host_profiles"
        notices = [r.getMessage() for r in caplog.records if "mixlocal" in r.getMessage()]
        assert len(notices) == 1 and "create_client" in notices[0]
        assert not any("mixhosted" in r.getMessage() for r in caplog.records)
    finally:
        host = getattr(plugins_mod.get_plugin_manager(), "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()
        for name in [m for m in sys.modules if m.endswith("_mixpair")]:
            sys.modules.pop(name, None)


@pytest.mark.platforms("any")
def test_a_rescan_keeps_the_hosted_plugin_it_already_loaded(tmp_path, monkeypatch, caplog):
    """Another plugin dir landing while the process runs rescans the home: the plugin already loaded
    is not imported again (no second warning, the same hosted profile object)."""
    import providers

    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "model-providers" / "mixpair"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: mixpair\nkind: model-provider\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(TWO_PROFILE_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"isolation": "host"}}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))

    try:
        with caplog.at_level(logging.WARNING, logger="providers"):
            first = providers.get_provider_profile("mixhosted")
            assert providers.get_provider_profile("mixlocal") is not None
            (home / "plugins" / "model-providers" / "other").mkdir()
            providers.refresh_bound_home()
            again = providers.get_provider_profile("mixhosted")
        assert first is not None and again is first
        assert len([r for r in caplog.records if "mixlocal" in r.getMessage()]) == 1
    finally:
        host = getattr(plugins_mod.get_plugin_manager(), "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()
        for name in [m for m in sys.modules if m.endswith("_mixpair")]:
            sys.modules.pop(name, None)
