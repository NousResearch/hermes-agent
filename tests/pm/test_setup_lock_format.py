"""Exercise setup's pre-Python pin reader through real downloads and extraction.

Only the downloaded uv executable and the PM command at the handoff are fixtures;
the copied setup script, curl, hashing, archive staging, and Python process are real.
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
from tests.pm.test_pm_core import make_tar, served  # noqa: F401 -- shared HTTP fixture


pytestmark = pytest.mark.platforms("posix")
REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("indent,blank_lines", [(2, False), (4, False), (0, False), ("\t", False), (4, True)],
                         ids=["two-spaces", "four-spaces", "no-indent", "tabs", "blank-lines"])
@pytest.mark.parametrize("mirror_indent", [2, 7], ids=["mirror-two-spaces", "mirror-seven-spaces"])
def test_setup_reads_pins_independent_of_indentation(tmp_path, served, indent, blank_lines, mirror_indent):
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
    calls = tmp_path / "uv-calls"
    receipt = core / "handoff.json"
    (core / "pm" / "__init__.py").touch()
    (core / "pm" / "cli.py").write_text(
        "import json, pathlib, sys\n"
        "assert sys.argv[1:] == ['install'], sys.argv\n"
        f"pathlib.Path({str(receipt)!r}).write_text(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    uv_script = (
        f"#!{bash}\nset -eu\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(calls))}\n"
        'case "$*" in\n'
        f'  --version) printf \'%s\\n\' "uv {uv_version}" ;;\n'
        f'  "python install --no-bin {py_version}") ;;\n'
        f'  "python find --managed-python {py_version}") printf \'%s\\n\' {shlex.quote(interpreter)} ;;\n'
        '  *) exit 91 ;;\nesac\n'
    )
    docroot, base_url = served
    filename, digest = make_tar(docroot, "uv.tar.gz", {"uv-fixture/uv": uv_script})
    artifact = {"sha256": digest, "url": f"{base_url}/{filename}"}
    decoy = {"sha256": "0" * 64, "url": f"{base_url}/wrong-target.tar.gz"}
    target = current_target()
    data = {"packages": {
        "before": {"artifacts": {target: decoy}, "version": "wrong-before"},
        "python": {"artifacts": {target: decoy}, "version": f"{py_version}.7+fixture"},
        "uv": {"artifacts": {"before-target": decoy, target: artifact, "after-target": decoy},
               "version": uv_version},
        "after": {"artifacts": {target: decoy}, "version": "wrong-after"},
    }}
    content = json.dumps(data, indent=indent)
    if blank_lines:
        content = content.replace("\n", "\n \t\n")
    (core / "pm" / "lock.json").write_text(content + "\n", encoding="utf-8")
    (core / "pm" / "artifact-mirror.json").write_text(
        json.dumps({"origin": base_url, "prefix": "mirror/"}, indent=mirror_indent), encoding="utf-8",
    )
    env = {"PATH": os.environ["PATH"], "HOME": str(home),
           "HERMES_HOME": str(home / ".hermes"), "HERMES_RUNTIME_DIR": str(runtime),
           "PYTHONNOUSERSITE": "1"}
    result = subprocess.run(
        [bash, str(core / "setup-hermes.sh"), "--runtime-only"], cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (runtime / f"uv-{uv_version}-{target}" / "uv").read_text() == uv_script
    assert json.loads(receipt.read_text()) == ["install"]
    assert calls.read_text().splitlines() == [
        "--version", f"python install --no-bin {py_version}",
        f"python find --managed-python {py_version}",
    ]
    assert not (home / ".local").exists()
    assert not (core / ".env").exists()
    assert not (home / ".hermes" / "skills").exists()
    print(f"runtime-only bootstrap: indent={indent!r}, blank_lines={blank_lines}: exit {result.returncode}")
    print(result.stdout)


def test_setup_mirror_fallback_reads_any_layout(tmp_path, served):
    """The mirror fallback consumes artifact-mirror.json read pre-Python.

    The primary URL is a dead port (curl exit 7, in the retriable set), so the
    staged artifact must come from the mirror; the mirror json is written as a
    compact single line to prove the reader follows keys, not indentation.
    """
    bash = shutil.which("bash")
    assert bash, "the shell bootstrap contract requires Bash"
    core = tmp_path / "checkout"
    (core / "pm").mkdir(parents=True)
    shutil.copy2(REPO / "setup-hermes.sh", core / "setup-hermes.sh")
    home = tmp_path / "home"
    home.mkdir()
    runtime = tmp_path / "runtime"
    interpreter = str(Path(sys._base_executable).resolve())
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    uv_version = "fixture-pin"
    receipt = core / "handoff.json"
    (core / "pm" / "__init__.py").touch()
    (core / "pm" / "cli.py").write_text(
        "import json, pathlib, sys\n"
        "assert sys.argv[1:] == ['install'], sys.argv\n"
        f"pathlib.Path({str(receipt)!r}).write_text(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    uv_script = (
        f"#!{bash}\nset -eu\n"
        f'case "$*" in\n'
        f'  --version) printf \'%s\\n\' "uv {uv_version}" ;;\n'
        f'  "python install --no-bin {py_version}") ;;\n'
        f'  "python find --managed-python {py_version}") printf \'%s\\n\' {shlex.quote(interpreter)} ;;\n'
        f'  *) exit 91 ;;\n'
        f'esac\n'
    )
    docroot, base_url = served
    mirror_dir = docroot / "mirror"
    mirror_dir.mkdir()
    filename, digest = make_tar(docroot, "uv.tar.gz", {"uv-fixture/uv": uv_script})
    shutil.copy2(docroot / filename, mirror_dir / digest)
    # Dead port: curl cannot connect (exit 7), which fetch_pinned treats as
    # retriable, forcing the mirror branch.
    dead = "http://127.0.0.1:1/uv.tar.gz"
    artifact = {"sha256": digest, "url": dead}
    target = current_target()
    data = {"packages": {
        "python": {"artifacts": {target: {"sha256": "0" * 64, "url": dead}},
                   "version": f"{py_version}.7+fixture"},
        "uv": {"artifacts": {target: artifact}, "version": uv_version},
    }}
    (core / "pm" / "lock.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    # Hostile indentation (one member per line, 9 spaces): the reader must
    # follow keys, not the 2-space layout the real file ships with.
    (core / "pm" / "artifact-mirror.json").write_text(
        json.dumps({"origin": base_url, "prefix": "mirror/"}, indent=9) + "\n", encoding="utf-8",
    )
    env = {"PATH": os.environ["PATH"], "HOME": str(home),
           "HERMES_HOME": str(home / ".hermes"), "HERMES_RUNTIME_DIR": str(runtime),
           "PYTHONNOUSERSITE": "1"}
    result = subprocess.run(
        [bash, str(core / "setup-hermes.sh"), "--runtime-only"], cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (runtime / f"uv-{uv_version}-{target}" / "uv").read_text() == uv_script
    assert json.loads(receipt.read_text()) == ["install"]
