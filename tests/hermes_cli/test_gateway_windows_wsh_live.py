"""Live Windows probe: Windows Script Host decodes the UTF-16 Startup chainer, so a target under a
non-ASCII directory is found and launched (W13, #143547).

Runs only on real Windows. It writes the chainer through the production ``_install_startup_entry``
under a temp APPDATA and runs it with ``cscript.exe``. No task or real Startup entry is touched.
"""

import subprocess

import pytest

from hermes_cli import gateway_windows
from tests.waiting import wait_until


@pytest.mark.platforms("windows")
def test_startup_chainer_launches_a_target_under_a_non_ascii_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    service_dir = tmp_path / "方舟 Usuário" / "gateway-service"
    service_dir.mkdir(parents=True)
    marker = service_dir / "ran.txt"
    script_path = service_dir / "gateway.cmd"
    target = f"CreateObject(\"Scripting.FileSystemObject\").CreateTextFile({gateway_windows._quote_vbs_string(str(marker))}).Close\r\n"
    script_path.with_suffix(".vbs").write_text(target, encoding="utf-16", newline="")

    entry = gateway_windows._install_startup_entry(script_path)
    run = subprocess.run(["cscript.exe", "//nologo", str(entry)], capture_output=True, text=True, timeout=60, check=False)
    assert run.returncode == 0, run.stdout + run.stderr

    # The chainer quits silently when FileExists(target) is false, so only a correctly decoded path
    # reaches the target; it is started asynchronously, hence the wait.
    wait_until(marker.exists, timeout=30, what="the Startup chainer to launch the target under the non-ASCII directory")
