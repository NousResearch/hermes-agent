"""setup-hermes.sh's bootstrap lookup survives an inherited uv Python preference.

uv reads ``UV_PYTHON_PREFERENCE`` as ``--python-preference`` and rejects it
beside ``--managed-python`` (0.12.13 and 0.13.x alike), so a shell exporting
it made ``uv python find --managed-python`` exit 2 and setup, ``source
./activate`` and ``scripts/run_tests.sh`` all died before PM ran. CI never
exports it, which is why the suite stayed green there. Only that one lookup
drops the variable: PM's handoff still sees the caller's preference.

The uv executable and the PM command at the handoff are fixtures; the copied
setup script and the Bash process are real.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from pm.store import current_target


pytestmark = pytest.mark.platforms("posix")
REPO = Path(__file__).resolve().parents[2]


def test_bootstrap_lookup_drops_inherited_python_preference(tmp_path):
    bash = shutil.which("bash")
    assert bash, "the shell bootstrap contract requires Bash"
    core = tmp_path / "checkout with spaces"
    (core / "pm").mkdir(parents=True)
    shutil.copy2(REPO / "setup-hermes.sh", core / "setup-hermes.sh")
    home = tmp_path / "home"
    home.mkdir()
    runtime = tmp_path / "runtime"
    interpreter = str(Path(sys._base_executable).resolve())
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    uv_version = "fixture-pin"
    target = current_target()
    calls = tmp_path / "uv-calls"
    receipt = core / "handoff.json"
    (core / "pm" / "__init__.py").touch()
    (core / "pm" / "cli.py").write_text(
        "import json, os, pathlib, sys\n"
        f"pathlib.Path({str(receipt)!r}).write_text(json.dumps("
        "{'argv': sys.argv[1:], 'preference': os.environ.get('UV_PYTHON_PREFERENCE')}))\n",
        encoding="utf-8",
    )
    # The real uv's behaviour, reduced to the one rule under test: clap treats
    # the environment variable as the flag it backs and rejects the pair.
    uv = runtime / f"uv-{uv_version}-{target}" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text(
        f"#!{bash}\nset -eu\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(calls))}\n"
        'case "$*" in\n'
        f'  --version) printf \'%s\\n\' "uv {uv_version}" ;;\n'
        f'  "python install --no-bin --no-registry {py_version}") ;;\n'
        f'  "python find --managed-python {py_version}")\n'
        '    if [ -n "${UV_PYTHON_PREFERENCE:-}" ]; then\n'
        "      echo 'error: the argument `--managed-python` cannot be used with `--python-preference`' >&2\n"
        "      exit 2\n"
        "    fi\n"
        f"    printf '%s\\n' {shlex.quote(interpreter)} ;;\n"
        '  *) exit 91 ;;\nesac\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    decoy = {"sha256": "0" * 64, "url": "http://127.0.0.1:1/wrong.tar.gz"}
    (core / "pm" / "lock.json").write_text(json.dumps({"packages": {
        "python": {"artifacts": {target: decoy}, "version": f"{py_version}.7+fixture"},
        "uv": {"artifacts": {target: decoy}, "version": uv_version},
    }}, indent=2) + "\n", encoding="utf-8")
    (core / "pm" / "artifact-mirror.json").write_text(
        json.dumps({"origin": "http://127.0.0.1:1", "prefix": "mirror/"}), encoding="utf-8",
    )
    env = {"PATH": os.environ["PATH"], "HOME": str(home),
           "HERMES_HOME": str(home / ".hermes"), "HERMES_RUNTIME_DIR": str(runtime),
           "PYTHONNOUSERSITE": "1", "UV_PYTHON_PREFERENCE": "only-managed"}
    result = subprocess.run(
        [bash, str(core / "setup-hermes.sh"), "--runtime-only"], cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls.read_text().splitlines() == [
        "--version", f"python install --no-bin --no-registry {py_version}",
        f"python find --managed-python {py_version}",
    ]
    # The scrub is scoped to the lookup: PM still inherits the caller's preference.
    assert json.loads(receipt.read_text()) == {
        "argv": ["install", "--test-environment", "--trust-recorded"],
        "preference": "only-managed",
    }
