"""Updater must not rewrite a user-owned Windows Scheduled Task action."""
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.platforms('windows')


def test_update_preserves_custom_task_and_launchers(monkeypatch, tmp_path):
    from hermes_cli import gateway_windows, update_cmd_windows

    script = tmp_path / 'gateway.cmd'
    script.write_bytes(b'user-owned launcher\r\n')
    registered = '''<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
      <Actions><Exec><Command>wscript.exe</Command>
      <Arguments>//B //Nologo "C:\\launchers\\supervisor.vbs"</Arguments>
      </Exec></Actions></Task>'''
    main = MagicMock()
    main._is_windows.return_value = True
    monkeypatch.setattr('hermes_cli.update_cmd._m', lambda: main)
    monkeypatch.setattr(gateway_windows, 'is_installed', lambda: True)
    monkeypatch.setattr(gateway_windows, 'is_task_registered', lambda: True)
    monkeypatch.setattr(gateway_windows, 'get_task_name', lambda: 'Hermes_Test')
    monkeypatch.setattr(gateway_windows, 'get_task_script_path', lambda: script)
    monkeypatch.setattr(gateway_windows, '_query_scheduled_task_xml', lambda name: registered)
    write = MagicMock()
    repair = MagicMock()
    cleanup = MagicMock(return_value=([], []))
    monkeypatch.setattr(gateway_windows, '_write_task_script', write)
    monkeypatch.setattr(gateway_windows, 'reconcile_scheduled_task', repair)
    monkeypatch.setattr(gateway_windows, 'reconcile_autostart_launchers', cleanup)

    update_cmd_windows._refresh_windows_gateway_launchers()

    write.assert_not_called()
    repair.assert_not_called()
    cleanup.assert_not_called()
    assert script.read_bytes() == b'user-owned launcher\r\n'


def test_update_preserves_registered_settings_with_native_action(monkeypatch, tmp_path):
    from hermes_cli import gateway_windows, update_cmd_windows

    script = tmp_path / 'gateway.cmd'
    registered = gateway_windows._build_scheduled_task_xml(
        'Hermes_Test', script.with_suffix('.vbs'), 'TEST\\user'
    ).replace(f'<Count>{gateway_windows._TASK_RESTART_COUNT}</Count>', '<Count>7</Count>')
    assert '<Count>7</Count>' in registered
    main = MagicMock()
    main._is_windows.return_value = True
    monkeypatch.setattr('hermes_cli.update_cmd._m', lambda: main)
    monkeypatch.setattr(gateway_windows, 'is_installed', lambda: True)
    monkeypatch.setattr(gateway_windows, 'is_task_registered', lambda: True)
    monkeypatch.setattr(gateway_windows, 'get_task_name', lambda: 'Hermes_Test')
    monkeypatch.setattr(gateway_windows, 'get_task_script_path', lambda: script)
    monkeypatch.setattr(gateway_windows, '_query_scheduled_task_xml', lambda name: registered)
    write = MagicMock()
    repair = MagicMock()
    monkeypatch.setattr(gateway_windows, '_write_task_script', write)
    monkeypatch.setattr(gateway_windows, 'reconcile_scheduled_task', repair)
    monkeypatch.setattr(gateway_windows, 'reconcile_autostart_launchers', lambda: ([], []))

    update_cmd_windows._refresh_windows_gateway_launchers()

    write.assert_called_once()
    repair.assert_not_called()


@pytest.mark.parametrize('definition', [None, '', '<broken', '<Task><Actions/></Task>'])
def test_unknown_task_definition_is_preserved(monkeypatch, tmp_path, definition):
    from hermes_cli import gateway_windows, update_cmd_windows

    main = MagicMock()
    main._is_windows.return_value = True
    monkeypatch.setattr('hermes_cli.update_cmd._m', lambda: main)
    monkeypatch.setattr(gateway_windows, 'is_installed', lambda: True)
    monkeypatch.setattr(gateway_windows, 'is_task_registered', lambda: True)
    monkeypatch.setattr(gateway_windows, 'get_task_name', lambda: 'Hermes_Test')
    monkeypatch.setattr(gateway_windows, 'get_task_script_path', lambda: tmp_path / 'gateway.cmd')
    monkeypatch.setattr(gateway_windows, '_query_scheduled_task_xml', lambda name: definition)
    write = MagicMock()
    repair = MagicMock()
    cleanup = MagicMock()
    monkeypatch.setattr(gateway_windows, '_write_task_script', write)
    monkeypatch.setattr(gateway_windows, 'reconcile_scheduled_task', repair)
    monkeypatch.setattr(gateway_windows, 'reconcile_autostart_launchers', cleanup)
    update_cmd_windows._refresh_windows_gateway_launchers()
    write.assert_not_called()
    repair.assert_not_called()
    cleanup.assert_not_called()


@pytest.mark.parametrize('discovery', ['query_error', 'second_query_error'])
def test_discovery_errors_do_not_bypass_preservation(monkeypatch, tmp_path, discovery):
    from hermes_cli import gateway_windows, update_cmd_windows
    main = MagicMock()
    main._is_windows.return_value = True
    monkeypatch.setattr('hermes_cli.update_cmd._m', lambda: main)
    monkeypatch.setattr(gateway_windows, 'is_startup_entry_installed', lambda: True)
    statuses = iter([True, False] if discovery == 'second_query_error' else [False, False])
    monkeypatch.setattr(gateway_windows, 'is_task_registered', lambda: next(statuses, False))
    monkeypatch.setattr(gateway_windows, 'get_task_name', lambda: 'Hermes_Test')
    monkeypatch.setattr(gateway_windows, '_query_scheduled_task_xml', lambda name: None)
    write = MagicMock()
    cleanup = MagicMock(return_value=([], []))
    monkeypatch.setattr(gateway_windows, '_write_task_script', write)
    monkeypatch.setattr(gateway_windows, 'reconcile_autostart_launchers', cleanup)
    update_cmd_windows._refresh_windows_gateway_launchers()
    write.assert_not_called()
    cleanup.assert_not_called()


def test_refresh_does_not_use_boolean_discovery(monkeypatch):
    from hermes_cli import gateway_windows, update_cmd_windows
    main = MagicMock()
    main._is_windows.return_value = True
    monkeypatch.setattr('hermes_cli.update_cmd._m', lambda: main)
    forbidden = MagicMock(side_effect=AssertionError('Boolean discovery must not be used'))
    monkeypatch.setattr(gateway_windows, 'is_installed', forbidden)
    monkeypatch.setattr(gateway_windows, 'is_task_registered', forbidden)
    monkeypatch.setattr(gateway_windows, 'get_task_name', lambda: 'Hermes_Test')
    inspect = MagicMock(return_value=None)
    write = MagicMock()
    monkeypatch.setattr(gateway_windows, '_query_scheduled_task_xml', inspect)
    monkeypatch.setattr(gateway_windows, '_write_task_script', write)
    update_cmd_windows._refresh_windows_gateway_launchers()
    forbidden.assert_not_called()
    inspect.assert_called_once_with('Hermes_Test')
    write.assert_not_called()


@pytest.mark.parametrize('customization', ['executable', 'extra_action', 'working_directory'])
def test_action_customizations_are_not_managed(monkeypatch, tmp_path, customization):
    from xml.etree import ElementTree
    from hermes_cli import gateway_windows

    script = tmp_path / 'gateway.cmd'
    root = ElementTree.fromstring(gateway_windows._build_scheduled_task_xml(
        'Hermes_Test', script.with_suffix('.vbs'), 'TEST'
    ))
    actions = next(e for e in root if e.tag.rsplit('}', 1)[-1] == 'Actions')
    if customization == 'executable':
        command = next(e for e in actions[0] if e.tag.rsplit('}', 1)[-1] == 'Command')
        command.text = 'custom-supervisor.exe'
    elif customization == 'extra_action':
        ElementTree.SubElement(actions, 'Exec')
    else:
        ElementTree.SubElement(actions[0], 'WorkingDirectory').text = 'C:\\custom'
    registered = ElementTree.tostring(root, encoding='unicode')
    monkeypatch.setattr(gateway_windows, 'get_task_script_path', lambda: script)
    monkeypatch.setattr(gateway_windows, '_query_scheduled_task_xml', lambda name: registered)
    assert gateway_windows.scheduled_task_uses_managed_action('Hermes_Test') is False
