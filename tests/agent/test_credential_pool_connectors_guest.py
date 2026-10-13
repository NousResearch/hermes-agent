"""A guest that does not carry inference here never serves inference through the credential pool.

A guest that claimed the welcome trial on the desktop's local backend leaves a ``device_code`` row
in ``credential_pool.nous``. In a process that is not that backend (plain CLI, gateway) the pool must
drop that row instead of handing its welcome-host token to the runtime.
"""

from hermes_cli import anon_auth
from tests.hermes_cli.anon_portal import claimed_trial, install_portal


def test_guest_pool_row_is_dropped_outside_the_desktop_backend(monkeypatch, tmp_path):
    from agent.credential_pool import load_pool
    from hermes_cli.auth_nous import resolve_nous_runtime_credentials

    install_portal(monkeypatch, tmp_path)
    claimed_trial(monkeypatch)
    resolve_nous_runtime_credentials()  # the claimed guest now carries an inference token
    assert any(e.source == "device_code" for e in load_pool("nous").entries())

    monkeypatch.delenv("HERMES_DESKTOP")

    assert not any(e.source == "device_code" for e in load_pool("nous").entries())
    assert anon_auth.has_guest()
