"""Automatic refresh must not replace a separately managed runtime selector stack."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from hermes_cli import gateway

@pytest.mark.linux_only
@pytest.mark.parametrize('system', [False, True])
def test_refresh_preserves_managed_runtime_and_all_original_bytes(tmp_path, monkeypatch, system):
    unit = tmp_path / 'hermes-gateway.service'
    original = '[Service]\nExecStart=/accepted/bin/python -m hermes_cli.main gateway run\nEnvironmentFile=/managed/private.env\nRestart=always\n'
    unit.write_text(original)
    dropins = unit.with_name(unit.name + '.d'); dropins.mkdir()
    rollback = dropins / '10-rollback.conf'
    rollback.write_text('[Service]\nExecStart=\nExecStart=/rollback/bin/python -m hermes_cli.main gateway run\n')
    active = dropins / '20-active.conf'
    active.write_text('[Service]\nExecStart=\nExecStart=/candidate/bin/python -m hermes_cli.main gateway run\nWorkingDirectory=/candidate/source\n')
    before = {p: p.read_bytes() for p in (unit, rollback, active)}
    calls = []
    monkeypatch.setattr(gateway, 'get_systemd_unit_path', lambda system=False: unit)
    monkeypatch.setattr(gateway, '_sync_hermes_home_from_systemd_unit', lambda **kw: None)
    monkeypatch.setattr(gateway, 'generate_systemd_unit', lambda **kw: '[Service]\nExecStart=/candidate/bin/python -m hermes_cli.main gateway run\n')
    monkeypatch.setattr(gateway, '_run_systemctl', lambda *a, **kw: calls.append(a) or SimpleNamespace(returncode=0))
    gateway.refresh_systemd_unit_if_needed(system=system)
    assert {p: p.read_bytes() for p in before} == before
    assert not calls

@pytest.mark.linux_only
@pytest.mark.parametrize('extra', ['', '[Unit]\nExecStart=/not-a-service-directive\n'])
def test_stock_refresh_still_updates_restart_contract_with_non_runtime_dropin(tmp_path, monkeypatch, extra):
    unit = tmp_path / 'hermes-gateway.service';unit.write_text('[Service]\nRestart=on-failure\n')
    dropins=unit.with_name(unit.name+'.d');dropins.mkdir()
    env=dropins/'environment.conf';env.write_text('[Service]\nEnvironment=EXTRA=1\n# ExecStart=comment only\n')
    (dropins / 'other.conf').write_text(extra)
    monkeypatch.setattr(gateway,'get_systemd_unit_path',lambda system=False:unit)
    monkeypatch.setattr(gateway,'generate_systemd_unit',lambda **kw:'[Service]\nRestart=always\n')
    calls=[]
    monkeypatch.setattr(gateway,'_run_systemctl',lambda *a,**kw:calls.append(a) or SimpleNamespace(returncode=0))
    assert gateway.refresh_systemd_unit_if_needed() is True
    assert unit.read_text()=='[Service]\nRestart=always\n'
    assert calls==[(['daemon-reload'],)]
    assert env.read_text().endswith('# ExecStart=comment only\n')

@pytest.mark.linux_only
def test_managed_system_refresh_keeps_required_home_sync(tmp_path, monkeypatch):
    unit = tmp_path / 'hermes-gateway-profile.service'
    unit.write_text('[Service]\nExecStart=/accepted/python\n')
    dropins = unit.with_name(unit.name + '.d'); dropins.mkdir()
    (dropins / 'override.conf').write_text('[Service]\nExecStart=\nExecStart=/managed/python\n')
    monkeypatch.setattr(gateway, 'get_systemd_unit_path', lambda system=False: unit)
    synced = []
    monkeypatch.setattr(gateway, '_sync_hermes_home_from_systemd_unit', lambda **kw: synced.append(kw))
    assert gateway.refresh_systemd_unit_if_needed(system=True) is False
    assert synced == [{'system': True}]

