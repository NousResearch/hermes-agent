"""Explicit requested-root naming and start-response encoding contracts."""
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize('named', [False, True])
def test_requested_service_name_does_not_borrow_callers_home(tmp_path, monkeypatch, named):
    from hermes_cli.gateway_runtime_service import service_suffix
    from hermes_cli.gateway import _profile_suffix
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    requested = tmp_path / 'custom'
    if named:
        requested = requested / 'profiles' / 'worker'
    requested.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(requested))
    installed = _profile_suffix()
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / '.hermes'))
    assert service_suffix(requested) == installed


@pytest.mark.linux_only
def test_service_start_does_not_decode_irrelevant_supervisor_output():
    from hermes_cli.gateway_runtime_service import ExistingService, start_existing_gateway_service
    service = ExistingService('systemd', (sys.executable, '-c', 'import sys; sys.stdout.buffer.write(bytes([255]))'))
    start_existing_gateway_service(service, deadline=time.monotonic() + 5)


def test_malformed_installed_plist_has_a_bounded_refusal():
    import plistlib
    from hermes_cli.gateway_runtime_service import _verify_binding, RuntimeStartError
    with pytest.raises(RuntimeStartError, match='service_identity_unverified'):
        _verify_binding(plistlib.loads, b'<?xml version="1.0"?><plist><dict>')
    definition = {'Label': 'ai.hermes.gateway'}
    assert _verify_binding(plistlib.loads, plistlib.dumps(definition)) == definition
