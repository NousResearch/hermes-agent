"""Native installer output must remain observable while the child is still alive."""

import base64
from pathlib import Path
import queue
import shutil
import subprocess
import threading

import pytest


def _literal(value):
    return "'" + str(value).replace("'", "''") + "'"


@pytest.mark.windows_only
@pytest.mark.parametrize("host", ["powershell.exe", "pwsh.exe"])
def test_native_output_is_delivered_before_child_can_complete(tmp_path, host):
    executable = shutil.which(host)
    if not executable:
        pytest.skip(f"{host} unavailable")
    installer = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"
    acknowledge = tmp_path / "output-observed"
    payload = (
        "[Console]::Out.WriteLine('EARLY_NATIVE_OUTPUT'); "
        f"while (-not (Test-Path -LiteralPath {_literal(acknowledge)})) "
        "{ Start-Sleep -Milliseconds 50 }; exit 0"
    )
    encoded = base64.b64encode(payload.encode("utf-16-le")).decode("ascii")
    script = tmp_path / "stream-fixture.ps1"
    script.write_text(
        f". {_literal(installer)} -HermesHome {_literal(tmp_path / 'home')} "
        f"-InstallDir {_literal(tmp_path / 'repo')}\n"
        f"$result = Invoke-ProcessWithWallClockTimeout -FilePath {_literal(executable)} "
        f"-ArgumentList @('-NoProfile', '-EncodedCommand', '{encoded}') -TimeoutSec 15\n"
        "if ($result.ExitCode -ne 0 -or $result.TimedOut) { exit 1 }\n"
        "[Console]::Out.WriteLine('COMMAND_COMPLETED')\n",
        encoding="utf-8-sig",
    )
    proc = subprocess.Popen(
        [executable, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    lines = queue.Queue()

    def read_output():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    captured = []
    try:
        while True:
            line = lines.get(timeout=25)
            assert line is not None, "".join(captured)
            captured.append(line)
            if "EARLY_NATIVE_OUTPUT" in line:
                break
        assert proc.poll() is None, "child completed before its output became visible"
        acknowledge.write_text("observed", encoding="ascii")
        assert proc.wait(timeout=15) == 0, "".join(captured)
        reader.join(timeout=5)
        while not lines.empty():
            line = lines.get_nowait()
            if line is not None:
                captured.append(line)
        assert "COMMAND_COMPLETED" in "".join(captured)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        proc.stdout.close()
