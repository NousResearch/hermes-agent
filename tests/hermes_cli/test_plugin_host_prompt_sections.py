"""plugins.isolation: host: a hosted system prompt section reads its session info as plain data."""
import pytest

from tests.hermes_cli.test_plugin_host import _home_with_plugins

SECTION_PLUGIN = '''
def register(ctx):
    ctx.register_system_prompt_section(
        "probe.info", lambda info: "session=" + info["session_id"] + " platform=" + info.get("platform", "?"))
'''


@pytest.mark.platforms('any')  # the host is a child process
@pytest.mark.parametrize('isolation', ['host', 'in_process'])
def test_a_prompt_section_reads_session_info_in_either_isolation_mode(tmp_path, monkeypatch, isolation):
    from hermes_cli.plugins import PluginManager
    _home_with_plugins(tmp_path, monkeypatch, {'infoprobe': SECTION_PLUGIN}, isolation=isolation)
    manager = PluginManager()
    manager.discover_and_load()
    try:
        assert manager._plugins['infoprobe'].error is None
        rendered = manager.render_system_prompt_sections({'session_id': 's1', 'platform': 'tui'})
        assert [section.content for section in rendered] == ['session=s1 platform=tui']
    finally:
        manager.unload()
        if isolation == 'host':
            manager._plugin_host().shutdown()
