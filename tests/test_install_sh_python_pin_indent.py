"""install.sh reads the python pin from pm/lock.json by structure, not layout.

bootstrap_python's pre-Python awk reader must follow object names and braces
(the same contract setup-hermes.sh's pin() established), so any indentation the
lock writer produces resolves the same pinned version.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.platforms("posix")


@pytest.mark.parametrize("indent,blank_lines", [(2, False), (4, False), (0, False), ("\t", False), (4, True)],
                         ids=["two-spaces", "four-spaces", "no-indent", "tabs", "blank-lines"])
def test_bootstrap_python_reads_pin_independent_of_indentation(tmp_path, indent, blank_lines):
    bash = shutil.which("bash")
    assert bash, "the shell bootstrap contract requires Bash"
    core = tmp_path / "checkout"
    (core / "pm").mkdir(parents=True)
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    interpreter = str(Path(sys._base_executable).resolve())
    calls = tmp_path / "uv-calls"
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{bash}\nset -eu\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(calls))}\n"
        'case "$*" in\n'
        f'  "python install --no-bin {py_version}") ;;\n'
        f'  "python find --managed-python {py_version}") printf \'%s\\n\' {shlex.quote(interpreter)} ;;\n'
        '  *) exit 91 ;;\n'
        'esac\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    decoy = {"sha256": "0" * 64, "url": "http://127.0.0.1:1/wrong.tar.gz"}
    data = {"packages": {
        "before": {"artifacts": {"any": decoy}, "version": "wrong-before"},
        "python": {"artifacts": {"any": decoy}, "version": f"{py_version}.7+fixture"},
        "after": {"artifacts": {"any": decoy}, "version": "wrong-after"},
    }}
    content = json.dumps(data, indent=indent)
    if blank_lines:
        content = content.replace("\n", "\n \t\n")
    (core / "pm" / "lock.json").write_text(content + "\n", encoding="utf-8")
    script = (
        f'source "{(ROOT / "scripts/install.sh").as_posix()}" --manifest\n'
        f'ensure_uv() {{ UV_CMD="{uv.as_posix()}"; }}\n'
        f'INSTALL_DIR="{core.as_posix()}"\n'
        "bootstrap_python\n"
    )
    env = dict(os.environ, HOME=str(tmp_path), HERMES_HOME=str(tmp_path / ".hermes"))
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    # A layout-sensitive reader resolves no version, falls back to "3.14", and
    # the fake uv exits 91 on the unexpected argument — so the recorded calls
    # pinning THIS interpreter's version are the contract.
    assert calls.read_text().splitlines() == [
        f"python install --no-bin {py_version}",
        f"python find --managed-python {py_version}",
    ]
