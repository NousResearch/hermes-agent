"""install.sh prerequisites: a git on PATH that cannot run is not git.

macOS ships /usr/bin/git as a stub until Apple's Command Line Tools are
installed, so `command -v git` passes and the clone fails much later with
"git clone failed; no checkout published".
"""

import os
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install.sh"
_STUB_ERR = (
    "xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), "
    "missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun"
)


def _exe(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run_prerequisites(tmp_path: Path, *, git_works: bool, system: str):
    bin_dir = tmp_path / "bin"
    log = tmp_path / "xcode-select.log"
    if git_works:
        _exe(bin_dir / "git", "#!/bin/sh\necho 'git version 2.50.1'\n")
    else:
        _exe(bin_dir / "git", f'#!/bin/sh\necho "{_STUB_ERR}" >&2\nexit 1\n')
    _exe(bin_dir / "curl", "#!/bin/sh\nexit 0\n")
    _exe(bin_dir / "uname", f"#!/bin/sh\necho {system}\n")
    _exe(bin_dir / "xcode-select", f'#!/bin/sh\necho "$*" >> "{log}"\nexit 1\n')
    probe = (
        'source "$1" --manifest\n'
        "NON_INTERACTIVE=true\n"
        "uv_bootstrap_target() { echo darwin-arm64; }\n"
        "stage_prerequisites\n"
    )
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", HOME=str(tmp_path))
    result = subprocess.run(
        ["bash", "-c", probe, "probe", str(_SCRIPT)],
        env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
    )
    return result, (log.read_text() if log.is_file() else "")


@pytest.mark.platforms("posix")
def test_macos_git_stub_fails_prerequisites_with_clt_remedy(tmp_path):
    result, xcode_calls = _run_prerequisites(tmp_path, git_works=False, system="Darwin")

    assert result.returncode != 0
    assert "xcode-select --install" in result.stderr
    assert "invalid active developer path" in result.stderr
    assert "prerequisites ok" not in result.stdout
    assert xcode_calls.split() == ["--install"]


@pytest.mark.platforms("posix")
def test_broken_git_elsewhere_names_git_error_without_xcode(tmp_path):
    result, xcode_calls = _run_prerequisites(tmp_path, git_works=False, system="Linux")

    assert result.returncode != 0
    assert "git is installed but cannot run" in result.stderr
    assert "xcode-select" not in result.stderr
    assert xcode_calls == ""


@pytest.mark.platforms("posix")
def test_working_git_passes_prerequisites(tmp_path):
    result, xcode_calls = _run_prerequisites(tmp_path, git_works=True, system="Darwin")

    assert result.returncode == 0, result.stderr
    assert "prerequisites ok" in result.stdout
    assert xcode_calls == ""
