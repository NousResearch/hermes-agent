"""The installer's bootstrap uv writes state under the Hermes root, not the user's (#101269).

uv's defaults (``~/.cache/uv``, ``~/.local/share/uv``) belong to the uv the user
installed themselves: a Hermes download landing there is visible to their
``uv python list``, removable by their ``uv python uninstall``, and fills a cache
they own. The Hermes-owned cache is ``pm.packages.uv_cache_dir()`` — the same
path ``UV_CACHE_DIR`` must name.
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

ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = ROOT / "scripts" / "install.sh"
pytestmark = pytest.mark.platforms("posix")


def _fake_uv(path: Path, *, bash: str, record: Path, boot_py: Path) -> None:
    """A uv that logs the state dirs it was given and answers the bootstrap ladder."""
    path.write_text(
        f"#!{bash}\n"
        'case "$1" in\n'
        '  --version) echo "uv 99.0.0"; exit 0 ;;\n'
        f'  python) printf "%s|%s\\n" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" '
        f'>> {shlex.quote(str(record))} ;;\n'
        'esac\n'
        'case "$1 $2" in\n'
        f'  "python find") printf "%s\\n" {shlex.quote(str(boot_py))} ;;\n'
        '  "python install") exit 0 ;;\n'
        'esac\n'
        'exit 0\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_bootstrap_python_pins_uv_state_under_hermes_home(tmp_path: Path) -> None:
    """Every uv call the bootstrap makes carries Hermes-owned cache and python dirs."""
    bash = shutil.which("bash")
    assert bash, "the shell bootstrap requires bash"
    home = tmp_path / "home" / ".hermes"
    checkout = tmp_path / "checkout"
    (checkout / "pm").mkdir(parents=True)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    (checkout / "pm" / "lock.json").write_text(
        json.dumps({"packages": {"python": {"version": version}}}, indent=2), encoding="utf-8"
    )

    record = tmp_path / "uv-state"
    boot_py = Path(sys._base_executable).resolve()
    assert boot_py.is_file()
    _fake_uv(tmp_path / "uv", bash=bash, record=record, boot_py=boot_py)

    env = {**os.environ, "HOME": str(tmp_path / "home"), "HERMES_HOME": str(home)}
    env.pop("HERMES_RUNTIME_DIR", None)
    # Source the real script for its functions, then run the real bootstrap.
    script = ('source "$1" --manifest; INSTALL_DIR="$2"; UV_CMD="$3"; bootstrap_python; '
              'printf "%s\\n" "$boot_py"')
    result = subprocess.run(
        [bash, "-c", script, "test", str(INSTALL_SH), str(checkout), str(tmp_path / "uv")],
        env=env, cwd=checkout, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == str(boot_py), result.stdout + result.stderr

    assert record.is_file(), "the bootstrap never invoked uv's python commands"
    for line in record.read_text(encoding="utf-8").splitlines():
        cache, python_dir = line.split("|")
        assert cache == str(home / "cache" / "uv"), line
        assert python_dir == str(home / "cache" / "uv-python"), line
