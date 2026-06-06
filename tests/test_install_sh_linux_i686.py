"""Behavioral regression tests for Linux i686 installer/runtime gates."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

import tomllib


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
SETUP_SH = REPO_ROOT / "setup-hermes.sh"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"


def _shell_function(path: Path, name: str) -> str:
    """Extract one top-level shell function for isolated behavior execution."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line == f"{name}() {{")
    for end in range(start + 1, len(lines)):
        if lines[end] == "}":
            return "\n".join(lines[start : end + 1])
    raise AssertionError(f"unterminated shell function: {name}")


def _run_bash(
    script: str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        cwd=cwd,
        env=merged,
        check=False,
    )


def _detector_script(
    path: Path,
    *,
    os_name: str,
    kernel: str,
    machine: str,
    bits: str | None,
) -> str:
    detector = _shell_function(path, "is_linux_i686")
    getconf = "return 1" if bits is None else f"printf '%s\\n' {shlex.quote(bits)}"
    os_assignment = f"OS={shlex.quote(os_name)}\n" if path == INSTALL_SH else ""
    return f"""
set -u
{os_assignment}{detector}
uname() {{
    case "${{1:-}}" in
        -s) printf '%s\\n' {shlex.quote(kernel)} ;;
        -m) printf '%s\\n' {shlex.quote(machine)} ;;
        *) return 1 ;;
    esac
}}
getconf() {{ {getconf}; }}
if is_linux_i686; then printf 'yes'; else printf 'no'; fi
"""


@pytest.mark.parametrize(
    ("os_name", "kernel", "machine", "bits", "expected"),
    [
        ("linux", "Linux", "i686", "32", "yes"),
        ("linux", "Linux", "i686", None, "yes"),
        ("linux", "Linux", "x86_64", "32", "yes"),
        ("linux", "Linux", "amd64", "32", "yes"),
        ("linux", "Linux", "x86_64", "64", "no"),
        ("linux", "Linux", "armv7l", "32", "no"),
        ("macos", "Darwin", "i686", "32", "no"),
    ],
)
def test_install_i686_detection_executes_userspace_logic(
    os_name: str,
    kernel: str,
    machine: str,
    bits: str | None,
    expected: str,
) -> None:
    result = _run_bash(
        _detector_script(
            INSTALL_SH,
            os_name=os_name,
            kernel=kernel,
            machine=machine,
            bits=bits,
        )
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


@pytest.mark.parametrize(
    ("kernel", "machine", "bits", "expected"),
    [
        ("Linux", "x86_64", "32", "yes"),
        ("Linux", "x86_64", "64", "no"),
        ("Linux", "armv7l", "32", "no"),
        ("Darwin", "i686", "32", "no"),
    ],
)
def test_checkout_setup_i686_detection_executes_userspace_logic(
    kernel: str,
    machine: str,
    bits: str,
    expected: str,
) -> None:
    result = _run_bash(
        _detector_script(
            SETUP_SH,
            os_name="linux",
            kernel=kernel,
            machine=machine,
            bits=bits,
        )
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


@pytest.mark.parametrize("path", [INSTALL_SH, SETUP_SH])
def test_dependency_profile_uses_linux_i686_for_32bit_userspace_under_x86_64(
    path: Path,
) -> None:
    detector = _shell_function(path, "is_linux_i686")
    selector = _shell_function(path, "dependency_profile")
    os_assignment = "OS=linux\n" if path == INSTALL_SH else ""
    result = _run_bash(
        f"""
set -u
{os_assignment}{detector}
{selector}
uname() {{
    case "${{1:-}}" in
        -s) printf 'Linux\n' ;;
        -m) printf 'x86_64\n' ;;
        *) return 1 ;;
    esac
}}
getconf() {{ printf '32\n'; }}
dependency_profile
"""
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "linux-i686\n"


def test_i686_temp_and_uv_directories_execute_under_hermes_home(tmp_path: Path) -> None:
    functions = "\n".join(
        _shell_function(INSTALL_SH, name)
        for name in (
            "is_linux_i686",
            "configure_linux_i686_tempdir",
            "configure_linux_i686_uv_python_dirs",
        )
    )
    script = f"""
set -eu
OS=linux
ROOT_FHS_LAYOUT=false
HERMES_HOME={shlex.quote(str(tmp_path / 'hermes'))}
unset TMPDIR UV_PYTHON_INSTALL_DIR UV_PYTHON_BIN_DIR
log_info() {{ :; }}
uname() {{ [ "$1" = -m ] && printf 'x86_64\\n'; }}
getconf() {{ printf '32\\n'; }}
{functions}
configure_linux_i686_tempdir
configure_linux_i686_uv_python_dirs
printf '%s\\n%s\\n%s\\n' "$TMPDIR" "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_BIN_DIR"
"""
    result = _run_bash(script)
    assert result.returncode == 0, result.stderr
    tmpdir, install_dir, bin_dir = result.stdout.splitlines()
    assert Path(tmpdir) == tmp_path / "hermes" / "tmp"
    assert Path(install_dir) == tmp_path / "hermes" / "uv" / "python"
    assert Path(bin_dir) == tmp_path / "hermes" / "uv" / "bin"
    assert all(Path(path).is_dir() for path in (tmpdir, install_dir, bin_dir))


def test_i686_temp_and_uv_directory_overrides_are_preserved(tmp_path: Path) -> None:
    functions = "\n".join(
        _shell_function(INSTALL_SH, name)
        for name in (
            "is_linux_i686",
            "configure_linux_i686_tempdir",
            "configure_linux_i686_uv_python_dirs",
        )
    )
    custom_tmp = tmp_path / "custom-tmp"
    custom_python = tmp_path / "custom-python"
    custom_bin = tmp_path / "custom-bin"
    script = f"""
set -eu
OS=linux
ROOT_FHS_LAYOUT=false
HERMES_HOME={shlex.quote(str(tmp_path / 'hermes'))}
TMPDIR={shlex.quote(str(custom_tmp))}
UV_PYTHON_INSTALL_DIR={shlex.quote(str(custom_python))}
UV_PYTHON_BIN_DIR={shlex.quote(str(custom_bin))}
export TMPDIR UV_PYTHON_INSTALL_DIR UV_PYTHON_BIN_DIR
log_info() {{ :; }}
uname() {{ [ "$1" = -m ] && printf 'i686\\n'; }}
getconf() {{ printf '32\\n'; }}
{functions}
configure_linux_i686_tempdir
configure_linux_i686_uv_python_dirs
printf '%s\\n%s\\n%s\\n' "$TMPDIR" "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_BIN_DIR"
"""
    result = _run_bash(script)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(custom_tmp),
        str(custom_python),
        str(custom_bin),
    ]


def _write_fake_python(path: Path) -> None:
    path.write_text(
        """#!/bin/sh
case "${1:-}" in
  -c) exit 0 ;;
  --version) printf 'Python 3.11.9\\n'; exit 0 ;;
  -m)
    if [ "${2:-}" = venv ]; then
      target="${3:-venv}"
      mkdir -p "$target/bin"
      cp "$0" "$target/bin/python"
      chmod +x "$target/bin/python"
      exit 0
    fi
    ;;
esac
exit 1
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_i686_uv_failure_executes_system_python_fallback(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3.11"
    _write_fake_python(fake_python)
    fake_uv = fake_bin / "uv"
    fake_uv.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    # Ensure the test exercises the Python 3.11 fallback even on CI images
    # that also provide a compatible system Python 3.12 or 3.13.
    for name in ("python3.13", "python3.12"):
        rejected = fake_bin / name
        rejected.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        rejected.chmod(0o755)

    functions = "\n".join(
        _shell_function(INSTALL_SH, name)
        for name in ("is_linux_i686", "find_compatible_python", "check_python")
    )
    script = f"""
set -u
OS=linux
DISTRO=linux
PYTHON_VERSION=3.11
UV_CMD={shlex.quote(str(fake_uv))}
LINUX_I686_SYSTEM_PYTHON=false
unset HERMES_PYTHON
PATH={shlex.quote(str(fake_bin))}:/usr/bin:/bin
export PATH
log_info() {{ :; }}
log_success() {{ :; }}
log_warn() {{ :; }}
log_error() {{ printf 'ERROR:%s\\n' "$*" >&2; }}
uname() {{ [ "$1" = -m ] && printf 'x86_64\\n'; }}
getconf() {{ printf '32\\n'; }}
{functions}
check_python
printf '%s|%s' "$LINUX_I686_SYSTEM_PYTHON" "$PYTHON_PATH"
"""
    result = _run_bash(script)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"true|{fake_python}"


def test_i686_system_python_path_executes_stdlib_venv(tmp_path: Path) -> None:
    fake_python = tmp_path / "python3.11"
    _write_fake_python(fake_python)
    setup_venv = _shell_function(INSTALL_SH, "setup_venv")
    script = f"""
set -eu
USE_VENV=true
DISTRO=linux
LINUX_I686_SYSTEM_PYTHON=true
PYTHON_PATH={shlex.quote(str(fake_python))}
INSTALL_DIR={shlex.quote(str(tmp_path))}
PYTHON_VERSION=3.11
UV_CMD=false
log_info() {{ :; }}
log_success() {{ :; }}
log_error() {{ printf 'ERROR:%s\\n' "$*" >&2; }}
{setup_venv}
setup_venv
printf '%s' "$UV_PYTHON"
"""
    result = _run_bash(script, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(tmp_path / "venv" / "bin" / "python")
    assert (tmp_path / "venv" / "bin" / "python").is_file()


def _selected(requirements: list[str], *, machine: str) -> list[Requirement]:
    env = dict(default_environment())
    env.update({"sys_platform": "linux", "platform_machine": machine})
    parsed = [Requirement(item) for item in requirements]
    return [req for req in parsed if req.marker is None or req.marker.evaluate(env)]


def _by_name(requirements: list[Requirement], name: str) -> list[Requirement]:
    return [req for req in requirements if req.name.lower() == name.lower()]


def test_core_and_linux_i686_profile_are_native_build_safe() -> None:
    data = tomllib.loads(PYPROJECT_TOML.read_text(encoding="utf-8"))
    core = _selected(data["project"]["dependencies"], machine="x86_64")
    pyjwt = _by_name(core, "PyJWT")
    uvicorn = _by_name(core, "uvicorn")
    assert len(pyjwt) == 1 and not pyjwt[0].extras
    assert not _by_name(core, "cryptography")
    assert len(uvicorn) == 1 and not uvicorn[0].extras

    web = data["project"]["optional-dependencies"]["web"]
    assert any(item == "uvicorn==0.41.0" for item in web)
    assert not any("uvicorn[standard]" in item for item in web)

    i686 = data["project"]["optional-dependencies"]["linux-i686"]
    selected_text = set(i686)
    assert not any("[crypto]" in item for item in selected_text)
    assert not any("[uvicorn-standard]" in item for item in selected_text)
    assert not any("[mcp]" in item for item in selected_text)
    assert not any("[google]" in item for item in selected_text)
    assert "hermes-agent[web]" in selected_text


def test_all_profile_keeps_pinned_crypto_and_server_accelerators_on_x86_64() -> None:
    data = tomllib.loads(PYPROJECT_TOML.read_text(encoding="utf-8"))
    optional = data["project"]["optional-dependencies"]
    assert set(optional["crypto"]) == {
        "PyJWT[crypto]==2.13.0",
        "cryptography==46.0.7",
    }
    assert optional["uvicorn-standard"] == ["uvicorn[standard]==0.41.0"]

    all_extra = _selected(optional["all"], machine="x86_64")
    selected_text = {str(req) for req in all_extra}
    assert any("[crypto]" in item for item in selected_text)
    assert any("[uvicorn-standard]" in item for item in selected_text)
    assert any("[mcp]" in item for item in selected_text)
    assert any("[google]" in item for item in selected_text)


def test_install_script_uses_portable_bash_shebang() -> None:
    assert INSTALL_SH.read_text(encoding="utf-8").splitlines()[0] == "#!/usr/bin/env bash"
