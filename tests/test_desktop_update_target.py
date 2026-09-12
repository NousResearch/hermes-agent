"""Run the real handoffs against a disposable CLI, never an installed updater."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "desktop-update"
FAKE_CLI = """
import json
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    if '--help' in sys.argv:
        print('update options')
        sys.exit(0)
    receipt = Path(os.environ['HANDOFF_CAPTURE'])
    previous = receipt.read_text(encoding='utf-8') if receipt.exists() else ''
    with receipt.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'argv': sys.argv[1:], 'home': os.environ.get('HERMES_HOME'),
                                 'install_root': os.environ.get('HERMES_INSTALL_ROOT'),
                                 'cwd': os.getcwd()}) + '\\n')
    sys.exit(1 if not previous else 0)
"""


def _run_handoff(tmp_path, target, *, windows=False, inherited_home=True):
    install = tmp_path / "checkout with spaces"
    if windows:
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(install / "venv")],
            check=True,
            capture_output=True,
            timeout=60,
        )
    package = (
        install / "venv" / "Lib" / "site-packages" if windows else install
    ) / "hermes_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "main.py").write_text(FAKE_CLI, encoding="utf-8")
    capture = tmp_path / "calls.jsonl"
    home = tmp_path / "profile home" if inherited_home else tmp_path
    home.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "HERMES_INSTALL_ROOT": str(install),
        "HANDOFF_CAPTURE": str(capture),
    }
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("HERMES_HOME", None)
    if inherited_home:
        env["HERMES_HOME"] = str(home)
    if windows:
        # The disposable runtime has only the fixture CLI; verification is outside
        # this transport contract and runs its own harmless fixture implementation.
        (package / "desktop_update_verify.py").write_text(
            "def verify_windows_desktop_update(): pass\n",
            encoding="utf-8",
        )
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPTS / "windows.ps1"),
            "-InstallRoot",
            str(install),
            "-NoUi",
        ]
    else:
        bin_dir = install / "venv" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python3").symlink_to(sys.executable)
        hermes = bin_dir / "hermes"
        hermes.write_text(
            f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} -m hermes_cli.main "$@"\n',
            encoding="utf-8",
        )
        hermes.chmod(0o755)
        command = [
            "bash",
            str(SCRIPTS / "posix.sh"),
            "--install-root",
            str(install),
            "--daemonized",
            "--no-ui",
        ]
    result = subprocess.run(
        [*command, *target],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    calls = (
        [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        if capture.exists()
        else []
    )
    return result, calls, home, install


def _assert_forwarded(
    tmp_path, target, expected, *, windows=False, inherited_home=True
):
    result, calls, home, install = _run_handoff(
        tmp_path,
        target,
        windows=windows,
        inherited_home=inherited_home,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 2, calls
    expected_args = ["update", "--yes", "--gateway"]
    if windows:
        expected_args += ["--force"]
    expected_args += expected
    assert (
        calls
        == [
            {
                "argv": expected_args,
                "home": str(home),
                "cwd": str(install),
                "install_root": str(install),
            }
        ]
        * 2
    )
    receipt = json.loads(
        (home / ".hermes-update-result.json").read_text(encoding="utf-8-sig")
    )
    assert receipt["ok"]
    assert receipt["channel"] == (expected[1] if expected[0] == "--channel" else "")
    assert not (home / ".hermes-update-in-progress").exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("channel", ["stable", "canary", "main"])
def test_posix_channel_survives_retry_in_active_profile(tmp_path, channel):
    _assert_forwarded(tmp_path, ["--channel", channel], ["--channel", channel])


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("channel", ["stable", "canary", "main"])
def test_windows_channel_survives_retry_in_active_profile(tmp_path, channel):
    _assert_forwarded(
        tmp_path, ["-Channel", channel], ["--channel", channel], windows=True
    )


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    "target, expected",
    [
        ([], ["--branch", "main"]),
        (["--branch", "feature/target"], ["--branch", "feature/target"]),
    ],
)
def test_posix_legacy_branch_and_default_home(tmp_path, target, expected):
    _assert_forwarded(tmp_path, target, expected, inherited_home=False)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize(
    "target, expected",
    [
        ([], ["--branch", "main"]),
        (["-Branch", "feature/target"], ["--branch", "feature/target"]),
    ],
)
def test_windows_legacy_branch_and_default_home(tmp_path, target, expected):
    _assert_forwarded(tmp_path, target, expected, windows=True, inherited_home=False)


def _assert_rejected(tmp_path, target, *, windows=False):
    result, calls, home, _ = _run_handoff(tmp_path, target, windows=windows)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not calls
    assert not (home / ".hermes-update-in-progress").exists()
    assert not (home / ".hermes-update-result.json").exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    "target",
    [
        ["--channel", "nightly"],
        ["--channel", ""],
        ["--channel"],
        ["--branch", "main", "--channel", "stable"],
        ["--channel", "canary", "--branch", "main"],
    ],
)
def test_posix_rejects_invalid_or_conflicting_target_before_update(tmp_path, target):
    _assert_rejected(tmp_path, target)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize(
    "target",
    [
        ["-Channel", "nightly"],
        ["-Channel", ""],
        ["-Channel"],
        ["-Branch", "main", "-Channel", "stable"],
        ["-Channel", "canary", "-Branch", "main"],
    ],
)
def test_windows_rejects_invalid_or_conflicting_target_before_update(tmp_path, target):
    _assert_rejected(tmp_path, target, windows=True)
