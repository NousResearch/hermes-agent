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
