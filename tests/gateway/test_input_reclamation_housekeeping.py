"""Ordinary housekeeping reaches only owned stores and never drains live aliases."""
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError


def _runner(homes):
    from gateway.session_authorities import SessionAuthorities

    registry = SessionAuthorities(homes[0])
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True),
        session_authorities=registry, _draining=False, _running=False,
        adapters={}, _profile_adapters={}, session_runtime_descriptor={'state': 'starting'})
    for index, home in enumerate(homes):
        home.mkdir(parents=True, exist_ok=True)
        authority = SimpleNamespace(db=SessionDB(home / 'state.db'),
            profile_id=str(home), epoch=index + 1, runner=runner)
        registry.add(home, authority, name=home.name)
    return runner


def _close(runner):
    for authority in runner.session_authorities:
        authority.db.close()


def test_housekeeping_scopes_each_owned_store_and_isolates_deferrals(tmp_path, monkeypatch):
    from gateway import run, hosted_room_input_reclamation as storage
    from gateway.run_input_reclamation import collect_gateway_input_copies
    from gateway.runtime_ownership import process_ownership
    from hermes_constants import get_hermes_home

    homes = [tmp_path / name for name in ('primary', 'secondary', 'unowned')]
    runner = _runner(homes)
    observed = []

    def collect(db, *, epoch, limit):
        observed.append((db.db_path, get_hermes_home(), epoch, limit))
        if epoch == 1:
            raise RuntimeStoreError('storage_unavailable')

    class FiveTicks:
        count = 0

        def is_set(self):
            return self.count == 5

        def wait(self, timeout=None):
            self.count += 1

    monkeypatch.setattr(storage, 'collect_working_copies', collect)
    monkeypatch.setattr(storage, 'collect_legacy_input_aliases',
        lambda *a, **k: pytest.fail('online housekeeping must not drain legacy aliases'))
    # Exercise the real housekeeping registration without running unrelated chores.
    monkeypatch.setattr(run, '_housekeeping_chore',
        lambda label, fn: fn() if label == 'Working-copy collection' else None)
    process_ownership.reserve(homes[:2])
    try:
        run._start_gateway_housekeeping(FiveTicks(), runner=runner, interval=0)
        assert [(str(path), home, epoch) for path, home, epoch, _ in observed] == [
            (str(home / 'state.db'), home, index + 1) for index, home in enumerate(homes[:2])]
        assert all(limit > 0 for _, _, _, limit in observed)
        runner._draining = True
        collect_gateway_input_copies(runner)
        assert len(observed) == 2
    finally:
        process_ownership.close()
        _close(runner)


@pytest.mark.parametrize('boundary', [
    'pre-ingress', 'ready', 'running', 'draining', 'api', 'control',
    'adapter', 'secondary-adapter', 'unowned', 'foreign-db',
])
def test_legacy_alias_drain_requires_the_pre_ingress_owner(tmp_path, monkeypatch, boundary):
    from gateway import hosted_room_input_reclamation as storage
    from gateway.run_input_reclamation import collect_legacy_copies_before_ingress
    from gateway.runtime_ownership import process_ownership

    home = tmp_path / 'owned'
    runner = _runner([home])
    authority = runner.session_authorities.launch
    original_db = authority.db
    foreign = None
    calls = []
    initialized = []
    monkeypatch.setattr(storage, 'initialize_working_copies',
        lambda db, **kwargs: initialized.append((db, kwargs)))
    monkeypatch.setattr(storage, 'collect_legacy_input_aliases',
        lambda db, **kwargs: calls.append((db, kwargs)))
    if boundary != 'unowned':
        process_ownership.reserve([home])
    if boundary == 'ready':
        runner.session_runtime_descriptor['state'] = 'ready'
    if boundary in {'running', 'draining'}:
        setattr(runner, '_' + boundary, True)
    if boundary in {'api', 'control'}:
        setattr(runner, 'session_api' if boundary == 'api' else 'session_control_server', object())
    if boundary == 'adapter':
        runner.adapters['telegram'] = object()
    if boundary == 'secondary-adapter':
        runner._profile_adapters['secondary'] = {'telegram': object()}
    if boundary == 'foreign-db':
        foreign = SessionDB(tmp_path / 'foreign.db')
        authority.db = foreign
    try:
        collect_legacy_copies_before_ingress(runner, authority)
        assert bool(calls) == (boundary == 'pre-ingress')
        assert bool(initialized) == bool(calls)
        if calls:
            assert calls[0][0] is original_db
            assert calls[0][1]['epoch'] == authority.epoch
    finally:
        process_ownership.close()
        authority.db = original_db
        _close(runner)
        if foreign is not None:
            foreign.close()
