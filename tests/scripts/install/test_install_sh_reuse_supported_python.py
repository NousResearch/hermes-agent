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

ROOT = Path(__file__).resolve().parent.parent.parent.parent
INSTALL_SH = ROOT / "scripts/install.sh"
pytestmark = pytest.mark.platforms("posix")


@pytest.mark.parametrize("option", ["--branch", "--commit", "--dir", "--hermes-home", "--stage"])
@pytest.mark.parametrize("suffix", [[], ["--manifest"], [""]])
def test_missing_value_fails_before_any_install_work(option: str, suffix: list[str]) -> None:
    result = subprocess.run(["bash", str(INSTALL_SH), option, *suffix],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert result.stderr.strip() == f"{option} needs a value"
    assert not result.stdout


@pytest.mark.parametrize("flags, expected", [
    ([], ["-m", "pm.cli", "install"]),
    (["--skip-browser"], ["-m", "pm.cli", "install", "--without", "agent-browser"]),
    (["--no-playwright"], ["-m", "pm.cli", "install", "--without", "agent-browser"]),
])
def test_browser_skip_becomes_the_pm_opt_out(tmp_path: Path, flags: list[str], expected: list[str]) -> None:
    """The skip flag is PM's persisted opt-out, not a no-op and not --non-interactive."""
    bash = shutil.which("bash")
    assert bash
    record = tmp_path / "pm-argv"
    boot = tmp_path / "boot-python"
    boot.write_text(f'#!{bash}\nprintf "%s\\n" "$@" > {shlex.quote(str(record))}\n', encoding="utf-8")
    boot.chmod(0o755)
    # Source the real script, replace only the interpreter acquisition, and run
    # the real PM stage function.
    script = ('source "$1" "${@:4}" --manifest; INSTALL_DIR="$2"; FIXTURE_PY="$3"; '
              'bootstrap_python() { boot_py="$FIXTURE_PY"; }; bootstrap_pm; '
              'printf "noninteractive=%s\\n" "$NON_INTERACTIVE"')
    result = subprocess.run([bash, "-c", script, "test", str(INSTALL_SH), str(tmp_path), str(boot), *flags],
                            env={**os.environ, "HERMES_HOME": str(tmp_path / "home")},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert record.read_text(encoding="utf-8").splitlines() == expected
    assert "noninteractive=false" in result.stdout


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
    # UV_CMD hands discovery the host uv as-is: which uv qualifies is
    # ensure_uv's contract (test_install_sh_repository_stage.py), not this one.
    script = ('source "$1" --manifest; INSTALL_DIR="$2"; UV_CMD="$3"; bootstrap_python; '
              '"$boot_py" -I -c "import sys; print(sys.prefix == sys.base_prefix)"')
    result = subprocess.run([bash, "-c", script, "test", str(INSTALL_SH), str(core), str(bin_dir / "uv")],
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
        # New enough for the installer's pin check, so the failure under test
        # is the Python one, not a rejected uv.
        'if [ "$1" = --version ]; then echo "uv 99.0.0"; exit 0; fi\n'
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