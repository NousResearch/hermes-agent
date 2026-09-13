"""Bootstrap reuses a supported base Python and reports failures (#10778).

PM owns the exact application runtime; the shell only needs a supported
interpreter to enter PM, never the activated app environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import venv

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "scripts/install.sh"
pytestmark = pytest.mark.platforms("posix")


def _environment(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    core = tmp_path / "checkout"
    (core / "pm").mkdir(parents=True)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    (core / "pm/lock.json").write_text(json.dumps(
        {"packages": {"python": {"version": version}}}, indent=2), encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("awk", "cut", "uname"):
        real = shutil.which(tool)
        assert real, f"the shell bootstrap requires {tool}"
        (bin_dir / tool).symlink_to(real)
    env = {**os.environ, "HOME": str(tmp_path / "home"),
           "HERMES_HOME": str(tmp_path / "home/.hermes"),
           "HERMES_RUNTIME_DIR": str(tmp_path / "tools"),
           "UV_PYTHON_INSTALL_DIR": str(tmp_path / "managed-python"),
           "UV_CACHE_DIR": str(tmp_path / "uv-cache"), "UV_OFFLINE": "1",
           "PATH": str(bin_dir)}
    return core, bin_dir, env


@pytest.mark.parametrize("activated", [False, True])
def test_supported_base_python_is_reused_offline(tmp_path: Path, activated: bool) -> None:
    """Use real uv discovery with no managed Python and, optionally, an active venv."""
    uv = shutil.which("uv")
    bash = shutil.which("bash")
    assert uv and bash, "bootstrap integration requires uv and Bash"
    core, bin_dir, env = _environment(tmp_path)
    (bin_dir / "uv").symlink_to(uv)
    (bin_dir / "python3").symlink_to(Path(sys._base_executable).resolve())
    if activated:
        active = core / "venv"
        venv.EnvBuilder(with_pip=False).create(active)
        env["VIRTUAL_ENV"] = str(active)
        env["PATH"] = f"{active / 'bin'}{os.pathsep}{bin_dir}"
    # Source the real helper, then inspect the selected interpreter, not argv.
    script = ('source "$1" --manifest; INSTALL_DIR="$2"; bootstrap_python; '
              '"$boot_py" -I -c "import sys; print(sys.prefix == sys.base_prefix)"')
    result = subprocess.run([bash, "-c", script, "test", str(INSTALL_SH), str(core)],
                            env=env, cwd=core, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "True", result.stdout + result.stderr
    assert not Path(env["UV_PYTHON_INSTALL_DIR"]).exists()
    if activated:
        assert (active / "bin/python").exists()


@pytest.mark.parametrize("failure", ["install", "lookup", "missing", "broken"])
def test_bootstrap_failure_has_no_success_frame(tmp_path: Path, failure: str) -> None:
    core, bin_dir, env = _environment(tmp_path)
    bash = shutil.which("bash")
    assert bash
    attempted = tmp_path / "attempted"
    broken = bin_dir / "broken-python"
    broken.write_text(f"#!{bash}\nexit 9\n", encoding="utf-8")
    broken.chmod(0o755)
    selected = broken if failure == "broken" else tmp_path / "missing-python"
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{bash}\n"
        'if [ "$1 $2" = "python install" ]; then\n'
        f"  : > {shlex.quote(str(attempted))}\n"
        f"  exit {9 if failure == 'install' else 0}\nfi\n"
        f"[ -f {shlex.quote(str(attempted))} ] || exit 2\n"
        + ("exit 2\n" if failure == "lookup" else f"printf '%s\\n' {shlex.quote(str(selected))}\n"),
        encoding="utf-8",
    )
    uv.chmod(0o755)
    result = subprocess.run([bash, str(INSTALL_SH), "--dir", str(core),
                             "--stage", "venv", "--json", "--non-interactive"],
                            env=env, cwd=core, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, result.stdout + result.stderr
    frame = json.loads(result.stdout.splitlines()[-1])
    assert frame["stage"] == "venv" and frame["ok"] is False, frame
    assert "bootstrap Python ready" not in result.stdout