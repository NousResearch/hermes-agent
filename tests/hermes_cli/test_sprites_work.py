from unittest.mock import Mock

import pytest

from hermes_cli import sprites_work


def test_turn_lease_released_on_error_and_job_reader_remains_protected(monkeypatch):
    api = Mock(return_value={})
    monkeypatch.setattr(sprites_work, 'available', lambda: True)
    monkeypatch.setattr(sprites_work.sprites_api, 'request', api)
    with pytest.raises(ValueError):
        with sprites_work.active_work():
            assert api.call_args.args[0] == 'PUT'
            raise ValueError('turn failed')
    assert [c.args[0] for c in api.call_args_list] == ['PUT', 'DELETE']
    assert api.call_args_list[0].args[1] == api.call_args_list[1].args[1]
    api.reset_mock()
    def reader(session):
        assert api.call_args.args[0] == 'PUT'
        return session
    assert sprites_work.run_tracked(reader, 'job') == 'job'
    assert [c.args[0] for c in api.call_args_list] == ['PUT', 'DELETE']


def test_unprotected_turn_is_rejected_but_existing_job_is_still_reaped(monkeypatch):
    monkeypatch.setattr(sprites_work, 'available', lambda: True)
    monkeypatch.setattr(sprites_work.sprites_api, 'request', Mock(side_effect=OSError('unavailable')))
    with pytest.raises(RuntimeError, match='Cannot protect'):
        with sprites_work.active_work():
            pytest.fail('unprotected turn admitted')
    reader = Mock(return_value='reaped')
    assert sprites_work.run_tracked(reader, 'job') == 'reaped'
    reader.assert_called_once_with('job')


def test_slow_renewal_expires_instead_of_racing_release(monkeypatch):
    monkeypatch.setattr(sprites_work, 'available', lambda: True)
    api = Mock()
    monkeypatch.setattr(sprites_work.sprites_api, 'request', api)
    thread = Mock()
    thread.is_alive.return_value = True
    monkeypatch.setattr('agent.memory_provider.spawn_context_thread', lambda *args, **kwargs: thread)
    with sprites_work.active_work():
        pass
    assert [call.args[0] for call in api.call_args_list] == ['PUT']
    assert api.call_args.args[2] == {'expire': '90s'}
