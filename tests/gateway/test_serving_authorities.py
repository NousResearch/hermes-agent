"""Readers of every served ledger skip a slot that serves nothing.

``serving_authorities`` is the registry's own ``SessionAuthorities.serving()``: a reserved-but-unbuilt
(``None``) slot and a parked profile's retained (retiring) authority are not served, so startup
recovery and a webhook's cross-ledger readback never read them.
"""
import sqlite3
from types import SimpleNamespace

import pytest

from gateway.session_authorities import SessionAuthorities, serving_authorities
from hermes_state_runtime import RuntimeStoreError


class _Ledger:
    def __init__(self, rows=(), broken=False):
        self.rows, self.broken = list(rows), broken

    def _read_all(self, *_args):
        if self.broken:
            raise sqlite3.OperationalError('disk I/O error')
        return self.rows


def _registry(tmp_path):
    homes = [tmp_path / name for name in ('root', 'beta', 'gamma')]
    for home in homes:
        home.mkdir()
    launch = SimpleNamespace(profile_id=str(homes[0]), db=_Ledger())
    parked = SimpleNamespace(profile_id=str(homes[1]), db=_Ledger(broken=True), retiring=True)
    registry = SessionAuthorities(homes[0], multiplexed=True)
    registry.add(homes[0], launch)
    registry.add(homes[1], parked, name='beta')
    registry.add(homes[2], None, name='gamma')  # reserved, not built yet
    return registry, launch


def test_serving_authorities_skips_unbuilt_and_retiring_slots(tmp_path):
    registry, launch = _registry(tmp_path)
    assert serving_authorities(SimpleNamespace(session_authorities=registry)) == [launch]


def test_a_webhook_readback_never_reads_a_parked_store(tmp_path, monkeypatch):
    from gateway.platforms.webhook_delivery import retained_destination
    registry, _launch = _registry(tmp_path)
    runner = SimpleNamespace(session_authorities=registry)
    # A reply leg scoped to a profile this process does not serve searches every served ledger.
    monkeypatch.setattr('gateway.session_authorities.active_authority', lambda _runner: None)
    adapter = SimpleNamespace(gateway_runner=runner, _message_handler=None, _delivery_info={})
    with pytest.raises(RuntimeStoreError, match='not_found'):
        retained_destination(adapter, 'chat-1')
