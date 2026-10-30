"""The 1 s hosted-room watcher must not enter owner_scope for an already-installed service.

Re-entering the scope re-reads each profile's .env on the event loop; with a few hundred
multiplexed profiles that starved the gateway loop and managed workers missed their receipt
budget.
"""
import asyncio
from types import SimpleNamespace

import gateway.session_hosted_service as shs


def _run(authority, monkeypatch):
    entered = []

    class _Scope:
        def __enter__(self):
            entered.append(1)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(shs, 'owner_scope', lambda a: _Scope())
    runner = SimpleNamespace(session_control_server=None)
    asyncio.run(shs._ensure_hosted_service(runner, authority))
    return entered


def test_installed_service_skips_owner_scope(monkeypatch):
    svc = SimpleNamespace(_transport_installed=True)
    assert _run(SimpleNamespace(hosted_room_service=svc), monkeypatch) == []


def test_uninstalled_service_still_enters_owner_scope(monkeypatch):
    installed = []
    monkeypatch.setattr('gateway.session_hosted_transport.install_hosted_transport',
                        lambda *a, **k: installed.append(1))
    svc = SimpleNamespace(_transport_installed=False, attest=None)
    authority = SimpleNamespace(hosted_room_service=svc)
    assert _run(authority, monkeypatch) == [1]
    assert installed == [1] and svc._transport_installed is True
