"""The installer E2E's capture must survive the installed PM launcher's -I."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("packaged", [False, True])
def test_isolated_pm_launch_captures_only_final_electron_spawn(tmp_path, packaged):
    launch_argv = (
        [str(tmp_path / "apps/desktop/release/mac-arm64/Hermes.app/Contents/MacOS/Hermes")]
        if packaged else ["npm", "exec", "--", "electron", "."]
    )
    root = Path(__file__).resolve().parents[3]
    helper = root / "tests/install/e2e-assets/launch-capture/pm-launch.py"
    launcher = tmp_path / "hermes"
    app = tmp_path / "app.py"
    app.write_text(
        "import os, subprocess, sys\n"
        "assert sys.flags.isolated == 1\n"
        "assert os.environ.get('PYTHONPATH') is None\n"
        "assert sys.argv[1:] == ['desktop']\n"
        "subprocess.run(['npm', 'run', 'build'], check=True)\n"
        f"subprocess.run({launch_argv!r}, "
        "cwd=os.getcwd(), env={**os.environ, 'PRODUCT_SENTINEL': 'present'}, check=True)\n",
        encoding="utf-8",
    )
    # Query emulates the published --print-runtime-command interface. The
    # returned command executes under a REAL isolated Python, not a patched
    # subprocess; the fake npm command proves builds still pass through.
    code = ("import os, runpy; os.environ.pop('PYTHONPATH', None); "
            f"runpy.run_path({str(app)!r}, run_name='__main__')")
    command = [sys.executable, "-I", "-c", code, "desktop"]
    launcher.write_text(
        "#!/bin/sh\n"
        "[ \"$1\" = --print-runtime-command ] || exit 90\n"
        f"printf '%s\\n' {shlex.quote(json.dumps(command))}\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    fake_npm = tmp_path / "npm"
    fake_npm.write_text(
        "#!/bin/sh\n"
        "[ \"$1 $2 $3\" = 'run build ' ] || exit 89\n"
        f"touch {shlex.quote(str(tmp_path / 'build-ran'))}\n",
        encoding="utf-8",
    )
    fake_npm.chmod(0o755)
    evil = tmp_path / "evil"
    evil.mkdir()
    (evil / "sitecustomize.py").write_text(
        f"open({str(tmp_path / 'ambient-hook-ran')!r}, 'w').close()\n", encoding="utf-8",
    )
    spec = tmp_path / "launch.json"
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
           "PYTHONPATH": str(evil)}
    result = subprocess.run([sys.executable, "-I", str(helper), str(launcher), str(spec)],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "build-ran").is_file()
    assert not (tmp_path / "ambient-hook-ran").exists()
    shape = "packaged" if packaged else "source"
    assert spec.with_name(spec.name + ".captured").read_text() == shape
    captured = json.loads(spec.read_text())
    assert captured["argv"] == launch_argv
    assert captured["matchedShape"] == shape
    assert captured["cwd"] == str(tmp_path)
    assert captured["env"]["PRODUCT_SENTINEL"] == "present"
