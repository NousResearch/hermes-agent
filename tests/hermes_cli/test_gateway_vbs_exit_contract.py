"""Execute generated launchers against harmless children, never a gateway."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from hermes_cli import gateway_windows as gw


@pytest.mark.skipif(os.name != 'nt', reason='Windows Script Host contract')
@pytest.mark.parametrize('exit_code', [0, 7])
def test_generated_launcher_waits_and_returns_child_exit(monkeypatch, tmp_path, exit_code):
    monkeypatch.setattr(gw, '_resolve_detached_python', lambda _: (sys.executable, Path(sys.prefix), []))
    monkeypatch.setattr(gw, '_gateway_run_argv', lambda *_: [sys.executable, '-c', f'import time; time.sleep(1); raise SystemExit({exit_code})'])
    text = gw._build_gateway_vbs_script(sys.executable, str(tmp_path), str(tmp_path), '')
    script = tmp_path / 'launcher.vbs'
    script.write_bytes(text.encode('utf-8'))
    start = time.monotonic()
    result = subprocess.run(['cscript.exe', '//nologo', str(script)], capture_output=True, timeout=20)
    assert result.returncode == exit_code, result.stderr.decode(errors='replace')
    assert time.monotonic() - start >= 1


def test_startup_wrapper_remains_async():
    text = gw._build_startup_launcher(Path('gateway.cmd'))
    assert ', 0, False' in text
