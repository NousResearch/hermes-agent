from pathlib import Path
import subprocess
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install.sh"


def _make_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _run_bash(command: str, *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_install_script_is_valid_shell():
    result = subprocess.run(["bash", "-n", str(INSTALL_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_check_python_prefers_existing_compatible_python_over_download(tmp_path):
    fake_python = _make_executable(
        tmp_path / "python313",
        """#!/bin/bash
if [ "$1" = "--version" ]; then
    echo "Python 3.13.12"
    exit 0
fi
exit 0
""",
    )
    install_marker = tmp_path / "uv-install-called"
    fake_uv = _make_executable(
        tmp_path / "uv",
        f"""#!/bin/bash
set -e
if [ "$1" = "python" ] && [ "$2" = "find" ] && [ "$3" = "3.11" ]; then
    exit 1
fi
if [ "$1" = "python" ] && [ "$2" = "find" ] && [ "$3" = ">=3.11" ]; then
    echo "{fake_python}"
    exit 0
fi
if [ "$1" = "python" ] && [ "$2" = "install" ] && [ "$3" = "3.11" ]; then
    touch "{install_marker}"
    exit 0
fi
echo "unexpected uv invocation: $*" >&2
exit 2
""",
    )

    command = textwrap.dedent(
        f"""
        source "{INSTALL_SCRIPT}"
        UV_CMD="{fake_uv}"
        DISTRO="macos"
        check_python
        printf 'PYTHON_PATH=%s\\n' "$PYTHON_PATH"
        printf 'PYTHON_FOUND_VERSION=%s\\n' "$PYTHON_FOUND_VERSION"
        """
    )

    result = _run_bash(command)

    assert result.returncode == 0, result.stderr
    assert f"PYTHON_PATH={fake_python}" in result.stdout
    assert "PYTHON_FOUND_VERSION=Python 3.13.12" in result.stdout
    assert not install_marker.exists()


def test_setup_venv_uses_resolved_python_path(tmp_path):
    fake_python = _make_executable(
        tmp_path / "python313",
        """#!/bin/bash
if [ "$1" = "--version" ]; then
    echo "Python 3.13.12"
    exit 0
fi
exit 0
""",
    )
    uv_args = tmp_path / "uv-args.txt"
    fake_uv = _make_executable(
        tmp_path / "uv",
        f"""#!/bin/bash
set -e
printf '%s\\n' "$@" > "{uv_args}"
mkdir -p venv/bin
cat > venv/bin/python <<'EOF'
#!/bin/bash
echo "Python 3.13.12"
EOF
chmod +x venv/bin/python
""",
    )

    command = textwrap.dedent(
        f"""
        cd "{tmp_path}"
        source "{INSTALL_SCRIPT}"
        UV_CMD="{fake_uv}"
        DISTRO="macos"
        USE_VENV=true
        PYTHON_VERSION="3.11"
        PYTHON_PATH="{fake_python}"
        PYTHON_FOUND_VERSION="Python 3.13.12"
        setup_venv
        """
    )

    result = _run_bash(command, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert uv_args.read_text(encoding="utf-8").splitlines() == [
        "venv",
        "venv",
        "--python",
        str(fake_python),
    ]
