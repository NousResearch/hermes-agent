"""install.ps1 reads the update channel a checkout chose, as Hermes wrote it (#137714).

The Windows twin of test_install_sh_chosen_channel: the record comes from Hermes's
own writer, and the installer's pre-Python reader must find it by this install's id.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.platforms("windows")


@pytest.mark.parametrize("channel", ["stable", "main"])
def test_installer_reads_the_channel_hermes_recorded(tmp_path, monkeypatch, channel):
    from hermes_cli.update_channel import set_install_channel

    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")
    home = tmp_path / "home"
    install = home / "hermes-agent"
    install.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    set_install_channel(channel, install)
    script = (f". '{ROOT / 'scripts' / 'install.ps1'}' -InstallDir '{install}' -HermesHome '{home}'\n"
              "Write-Output \"channel=$(Get-ChosenChannel)\"\n")
    wrapper = tmp_path / "read.ps1"
    wrapper.write_text(script, encoding="utf-8")

    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(wrapper)],
        check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL, timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"channel={channel}" in result.stdout.splitlines(), result.stdout + result.stderr
