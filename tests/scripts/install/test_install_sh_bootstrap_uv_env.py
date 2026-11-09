"""install.sh's bootstrap_python survives an inherited uv Python preference.

uv reads ``UV_PYTHON_PREFERENCE`` as ``--python-preference`` and rejects it
beside ``--managed-python`` (0.12.13 and 0.13.x alike). With the variable
exported, the managed lookup failed silently, the ``--system`` fallback took
over, and the user's only-managed preference decided nothing. The lookup
drops the variable for itself only, so the managed interpreter is found
first time and no fallback or download runs.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.platforms("posix")


def test_bootstrap_python_drops_inherited_python_preference(tmp_path):
    bash = shutil.which("bash")
    assert bash, "the shell bootstrap contract requires Bash"
    core = tmp_path / "checkout"
    (core / "pm").mkdir(parents=True)
    py_version = "3.77"
    interpreter = str(Path(sys._base_executable).resolve())
    calls = tmp_path / "uv-calls"
    uv = tmp_path / "uv"
    # The real uv's behaviour, reduced to the one rule under test: clap treats
    # the environment variable as the flag it backs and rejects the pair. Any
    # other request (the --system fallback, a download) is a wrong turn.
    uv.write_text(
        f"#!{bash}\nset -eu\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(calls))}\n"
        'case "$*" in\n'
        f'  "python find --managed-python {py_version}")\n'
        '    if [ -n "${UV_PYTHON_PREFERENCE:-}" ]; then\n'
        "      echo 'error: the argument `--managed-python` cannot be used with `--python-preference`' >&2\n"
        "      exit 2\n"
        "    fi\n"
        f"    printf '%s\\n' {shlex.quote(interpreter)} ;;\n"
        '  *) exit 91 ;;\n'
        'esac\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    decoy = {"sha256": "0" * 64, "url": "http://127.0.0.1:1/wrong.tar.gz"}
    (core / "pm" / "lock.json").write_text(json.dumps({"packages": {
        "python": {"artifacts": {"any": decoy}, "version": f"{py_version}.7+fixture"},
    }}, indent=2) + "\n", encoding="utf-8")
    script = (
        f'source "{(ROOT / "scripts/install.sh").as_posix()}" --manifest\n'
        f'ensure_uv() {{ UV_CMD="{uv.as_posix()}"; }}\n'
        f'INSTALL_DIR="{core.as_posix()}"\n'
        "bootstrap_python\n"
        'printf \'%s\\n\' "$boot_py"\n'
        # The scrub is scoped to the lookup: the installer keeps the caller's preference.
        'printf \'%s\\n\' "${UV_PYTHON_PREFERENCE:-unset}"\n'
    )
    env = dict(os.environ, HOME=str(tmp_path), HERMES_HOME=str(tmp_path / ".hermes"),
               UV_PYTHON_PREFERENCE="only-managed")
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls.read_text().splitlines() == [f"python find --managed-python {py_version}"]
    assert result.stdout.splitlines()[-2:] == [interpreter, "only-managed"]
