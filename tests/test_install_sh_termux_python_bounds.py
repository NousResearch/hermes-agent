"""Behavioral regression tests for Termux Python selection."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
SETUP_HERMES_SH = REPO_ROOT / "setup-hermes.sh"


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _write_fake_python(bin_dir: Path, name: str, version: str) -> Path:
    return _write_executable(
        bin_dir / name,
        f"""#!{sys.executable}
import os
import sys

VERSION = {version!r}
VERSION_INFO = tuple(int(part) for part in VERSION.split('.')[:3]) + ('final', 0)

if len(sys.argv) >= 2 and sys.argv[1] == '--version':
    print(f'Python {{VERSION}}')
    raise SystemExit(0)

if len(sys.argv) >= 3 and sys.argv[1] == '-c':
    sys.version = f'{{VERSION}} (fake)'
    sys.version_info = VERSION_INFO
    exec(sys.argv[2], {{'__name__': '__main__'}})
    raise SystemExit(0)

if len(sys.argv) >= 3 and sys.argv[1:3] == ['-m', 'venv']:
    target = sys.argv[3] if len(sys.argv) >= 4 else 'venv'
    bin_path = os.path.join(target, 'bin')
    os.makedirs(bin_path, exist_ok=True)
    python_path = os.path.join(bin_path, 'python')
    with open(python_path, 'w', encoding='utf-8') as handle:
        handle.write('''#!/bin/sh\nif [ "${{1:-}}" = '-m' ] && [ "${{2:-}}" = 'pip' ]; then\n    exit 0\nfi\nexit 0\n''')
    os.chmod(python_path, 0o755)
    raise SystemExit(0)

raise SystemExit(0)
""",
    )


def _write_unsupported_explicit_pythons(bin_dir: Path, *except_names: str) -> None:
    for name in ("python3.11", "python3.12", "python3.13"):
        if name not in except_names and not (bin_dir / name).exists():
            _write_fake_python(bin_dir, name, "3.14.6")


def _write_termux_command_stubs(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "uname",
        "#!/bin/sh\n[ \"${1:-}\" = '-s' ] && echo Linux || echo Linux\n",
    )
    _write_executable(bin_dir / "pkg", "#!/bin/sh\nexit 0\n")
    _write_executable(bin_dir / "git", "#!/bin/sh\necho 'git version 2.50.0'\n")
    _write_executable(bin_dir / "node", "#!/bin/sh\necho 'v22.12.0'\n")
    _write_executable(bin_dir / "npm", "#!/bin/sh\nexit 0\n")
    _write_executable(bin_dir / "curl", "#!/bin/sh\nexit 0\n")
    _write_executable(bin_dir / "rg", "#!/bin/sh\nexit 0\n")


def _termux_env(tmp_path: Path, bin_dir: Path) -> dict[str, str]:
    prefix = tmp_path / "com.termux" / "files" / "usr"
    (prefix / "bin").mkdir(parents=True)
    env = os.environ.copy()
    env.update({
        "ANDROID_API_LEVEL": "35",
        "HOME": str(tmp_path / "home"),
        "HERMES_HOME": str(tmp_path / "home" / ".hermes"),
        "PATH": f"{bin_dir}{os.pathsep}{env.get('PATH', os.defpath)}",
        "PREFIX": str(prefix),
        "TERMUX_VERSION": "0.118.0",
    })
    return env


def _run_install_prerequisites(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_termux_command_stubs(bin_dir)
    env = _termux_env(tmp_path, bin_dir)
    bash = shutil.which("bash") or "/bin/bash"
    return subprocess.run(
        [bash, str(INSTALL_SH), "--stage", "prerequisites", "--non-interactive"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _copy_setup_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copy2(SETUP_HERMES_SH, checkout / "setup-hermes.sh")
    return checkout


def _run_setup(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_termux_command_stubs(bin_dir)
    env = _termux_env(tmp_path, bin_dir)
    checkout = _copy_setup_checkout(tmp_path)
    bash = shutil.which("bash") or "/bin/bash"
    return subprocess.run(
        [bash, str(checkout / "setup-hermes.sh")],
        env=env,
        input="n\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_install_stage_prefers_compatible_minor_over_unsupported_default(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_python(bin_dir, "python3.11", "3.11.15")
    _write_fake_python(bin_dir, "python", "3.14.6")

    result = _run_install_prerequisites(tmp_path)

    assert result.returncode == 0, result.stdout
    assert "Python found: Python 3.11.15" in result.stdout


def test_install_stage_rejects_post_install_unsupported_default(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_python(bin_dir, "python", "3.14.6")
    _write_unsupported_explicit_pythons(bin_dir)

    result = _run_install_prerequisites(tmp_path)

    assert result.returncode == 1
    assert "Termux Python Python 3.14.6 is not supported" in result.stdout
    assert "Hermes requires Python >=3.11,<3.14" in result.stdout
    assert (
        "Install a compatible Termux Python (for example python3.11)" in result.stdout
    )


def test_setup_script_prefers_compatible_minor_over_unsupported_default(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_python(bin_dir, "python3.11", "3.14.6")
    _write_fake_python(bin_dir, "python3.12", "3.12.11")
    _write_fake_python(bin_dir, "python", "3.14.6")

    result = _run_setup(tmp_path)

    assert result.returncode == 0, result.stdout
    assert "Python 3.12.11 found" in result.stdout
    assert (tmp_path / "checkout" / "venv" / "bin" / "python").exists()


def test_setup_script_rejects_unsupported_default(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_python(bin_dir, "python", "3.14.6")
    _write_unsupported_explicit_pythons(bin_dir)

    result = _run_setup(tmp_path)

    assert result.returncode == 1
    assert "Termux Python Python 3.14.6 is not supported" in result.stdout
    assert "Hermes requires Python >=3.11,<3.14" in result.stdout
