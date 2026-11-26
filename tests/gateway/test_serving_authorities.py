"""Readers of every served ledger skip a slot that serves nothing.

``serving_authorities`` is the registry's own ``SessionAuthorities.serving()``: a reserved-but-unbuilt
(``None``) slot and a parked profile's retained (retiring) authority are not served, so startup
recovery and a webhook's cross-ledger readback never read them.
"""
from types import SimpleNamespace

import pytest

from gateway.session_authorities import SessionAuthorities, serving_authorities
from persistence.core.errors import AuthorityRefusal
from persistence import TransientStoreError


class _Ledger:
    """A served authority's storage interface handle: its admission ledger, or a store that fails every read."""

    def __init__(self, rows=(), broken=False):
        self.rows, self.broken = list(rows), broken
        self.admissions = self

    def webhook_ingress_payloads(self, *, chat_id):
        if self.broken:
            raise TransientStoreError('disk I/O error')
        return self.rows


def _registry(tmp_path):
    homes = [tmp_path / name for name in ('root', 'beta', 'gamma')]
    for home in homes:
        home.mkdir()
    launch = SimpleNamespace(profile_id=str(homes[0]), persistence=_Ledger())
    parked = SimpleNamespace(profile_id=str(homes[1]), persistence=_Ledger(broken=True), retiring=True)
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
    with pytest.raises(AuthorityRefusal, match='not_found'):
        retained_destination(adapter, 'chat-1')
