"""WSL host setup must never install/repair the Linux guest (#114531)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import tools_config_cua as install
from hermes_cli.config import save_config
from tools.computer_use import cua_backend as cb


@pytest.mark.linux_only
@pytest.mark.parametrize('case', ['ready', 'repair', 'fresh', 'unattended', 'bad-override', 'bad-contract'])
def test_wsl_setup_requires_the_host_contract_and_autostart(tmp_path, monkeypatch, case):
    monkeypatch.setattr('hermes_constants.is_wsl', lambda: True)
    save_config({'computer_use': {'target': 'windows'}})
    exe = str(tmp_path / 'Windows' / "Alice O'Neil" / 'cua-driver.exe')
    state = {'binary': case in {'ready', 'repair', 'unattended'}, 'task': case == 'ready'}
    monkeypatch.setattr(install, '_resolved_cua_driver_cmd', lambda: exe if state['binary'] else None)
    monkeypatch.setattr(install, '_cua_driver_contract_status',
                        lambda *a: {'ready': state['binary'] and case != 'bad-contract'})
    which = install.shutil.which
    monkeypatch.setattr(install.shutil, 'which', lambda name: name if name in {exe, 'powershell.exe'} else which(name))
    monkeypatch.delenv('HERMES_CUA_DRIVER_CMD', raising=False)
    if case == 'bad-override':
        monkeypatch.setenv('HERMES_CUA_DRIVER_CMD', '/usr/bin/linux-driver')
    commands = []
    def run(argv, **kwargs):
        commands.append(argv)
        if argv[0] == 'schtasks.exe':
            assert argv[1:] == ['/Query', '/TN', 'cua-driver-serve']
            return SimpleNamespace(returncode=0 if state['task'] else 1)
        if argv[0] == 'wslpath':
            assert argv[1:] == ['-w', exe]
            return SimpleNamespace(returncode=0, stdout=r"C:\Users\Alice O'Neil\cua-driver.exe")
        assert argv[0] == 'powershell.exe', 'must not run the POSIX installer'
        if install._CUA_INSTALL_PS1_URL in argv[-1]:
            assert '-NoAutoStart' not in argv[-1]
            assert kwargs['stdin'] == install.subprocess.DEVNULL
            state['binary'] = state['task'] = True
        else:
            assert "'C:\\Users\\Alice O''Neil\\cua-driver.exe'" in argv[-1]
            assert "@('autostart','enable')" in argv[-1]
            state['task'] = True
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(install.subprocess, 'run', run)
    expected = case in {'ready', 'repair', 'fresh'}
    assert install.install_cua_driver(upgrade=case == 'unattended',
                                     require_confirmed_update=case == 'unattended') is expected
    assert install._cua_driver_install_ready() is expected
    writes = [argv for argv in commands if argv[0] == 'powershell.exe']
    assert bool(writes) == (case in {'repair', 'fresh', 'bad-contract'})
    # A runtime probe must not raise UAC or install across OSes implicitly.
    monkeypatch.delenv('HERMES_CUA_DRIVER_CMD', raising=False)
    monkeypatch.setattr(cb, '_contract_repair_attempted', False)
    before = len(commands)
    broken = {'binary': exe, 'ready': False, 'reason': 'old contract'}
    assert cb._maybe_repair_runtime_contract(broken) is broken
    assert len(commands) == before
