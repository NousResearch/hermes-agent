"""Regression coverage for the user-facing macOS Hermes launcher."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _make_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _setup_path_function() -> str:
    match = re.search(
        r"^setup_path\(\) \{\n.*?^}\n",
        INSTALL_SH.read_text(encoding="utf-8"),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "setup_path function not found in scripts/install.sh"
    return match.group(0)


def test_venv_launcher_bypasses_uv_console_script_that_requires_realpath(tmp_path: Path) -> None:
    """Stock macOS must start Hermes even when its uv console script needs realpath."""
    install_dir = tmp_path / "install"
    venv_bin = install_dir / "venv" / "bin"
    command_dir = tmp_path / "command"
    minimal_path = tmp_path / "minimal-path"
    result = tmp_path / "launch-result"
    venv_bin.mkdir(parents=True)
    minimal_path.mkdir()

    dirname = shutil.which("dirname")
    assert dirname is not None
    (minimal_path / "dirname").symlink_to(dirname)

    _make_executable(
        venv_bin / "python",
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$LAUNCH_RESULT"\n',
    )
    (install_dir / "hermes").write_text("# source entrypoint\n", encoding="utf-8")
    _make_executable(
        venv_bin / "hermes",
        "#!/bin/sh\n"
        f'PATH="{minimal_path}"\n'
        "'''exec' \"$(dirname -- \"$(realpath -- \"$0\")\")\"/'python3' \"$0\" \"$@\"\n"
        "' '''\n",
    )

    harness = "\n".join(
        [
            "set -e",
            'get_command_link_dir() { printf "%s" "$COMMAND_LINK_DIR"; }',
            'get_command_link_display_dir() { printf "%s" "$COMMAND_LINK_DIR"; }',
            "log_info() { :; }",
            "log_success() { :; }",
            _setup_path_function(),
            "setup_path",
        ]
    )
    env = os.environ | {
        "USE_VENV": "true",
        "INSTALL_DIR": str(install_dir),
        "DISTRO": "macos",
        "COMMAND_LINK_DIR": str(command_dir),
    }
    subprocess.run(["/bin/bash", "-c", harness], env=env, check=True)

    completed = subprocess.run(
        [command_dir / "hermes", "--version"],
        env=os.environ | {"LAUNCH_RESULT": str(result)},
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert result.read_text(encoding="utf-8").splitlines() == [
        str(install_dir / "hermes"),
        "--version",
    ]
