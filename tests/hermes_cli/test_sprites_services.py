from unittest.mock import Mock

from hermes_cli.sprites_services import SpritesServiceManager
from hermes_cli import sprites_api


def test_existing_stopped_service_is_never_recreated_or_started(monkeypatch):
    api = Mock(return_value={'name': 'gateway-default', 'state': {'status': 'stopped'}})
    monkeypatch.setattr(sprites_api, 'request', api)
    SpritesServiceManager().register_profile_gateway('default', start_now=True)
    api.assert_called_once_with('GET', '/services/gateway-default')


def test_deliberate_stop_uses_service_control_and_profile_paths_are_validated(monkeypatch):
    import pytest
    api = Mock()
    monkeypatch.setattr(sprites_api, 'request', api)
    manager = SpritesServiceManager()
    manager.stop('gateway-default')
    api.assert_called_once_with('POST', '/services/gateway-default/stop')
    with pytest.raises(ValueError):
        manager.register_profile_gateway('../default')
    assert api.call_count == 1


def test_stopped_intent_is_profile_scoped_and_hides_parked_native_service(tmp_path, monkeypatch):
    from hermes_cli.sprites_services import gateway_is_deliberately_stopped
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'gateway_state.json').write_text('{"gateway_state":"stopped"}', encoding='utf-8')
    assert not gateway_is_deliberately_stopped('gateway-default')
    for profile in ('a', 'b'):
        (tmp_path / 'profiles' / profile).mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profiles' / 'a'))
    api = Mock(return_value={'state': {'status': 'running'}})
    monkeypatch.setattr(sprites_api, 'request', api)
    manager = SpritesServiceManager()
    manager.stop('gateway-a')
    assert gateway_is_deliberately_stopped('gateway-a')
    assert not manager.is_running('gateway-a')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profiles' / 'b'))
    manager.start('gateway-b')
    assert not gateway_is_deliberately_stopped('gateway-b')
    assert gateway_is_deliberately_stopped('gateway-a')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profiles' / 'a'))
    manager.start('gateway-a')
    assert not gateway_is_deliberately_stopped('gateway-a')
    assert manager.is_running('gateway-a')
