"""Session toolset handles stay in the process that owns their registrations."""

import pytest

from hermes_cli.plugin_isolation_audit import audit_plugin_dir
from hermes_cli.plugin_session_toolsets import active_session_toolset_names
from hermes_cli.plugins import PluginManager
from tests.hermes_cli.test_plugin_host import _home_with_plugins


@pytest.mark.platforms("any")  # exercises the real plugin child process on each host OS
def test_host_rejects_session_toolsets_before_creating_registration_handles(tmp_path, monkeypatch):
    home = _home_with_plugins(tmp_path, monkeypatch, {
        "sessionprobe": '''
def register(ctx):
    ctx.session_toolset("agent:main:test:dm:owner", name="client-tools")
''',
    })
    manager = PluginManager()
    try:
        manager.discover_and_load()
        error = str(manager._plugins["sessionprobe"].error)
        assert "session_toolset" in error
        assert "cannot run in the plugin host" in error
        assert active_session_toolset_names(scope=manager.scope_key) == set()

        audit = audit_plugin_dir(home / "plugins" / "sessionprobe")
        assert audit.verdict == "in_process"
        assert any("session_toolset" in reason for reason in audit.reasons)
    finally:
        manager.unload()
        manager._plugin_host().shutdown()
