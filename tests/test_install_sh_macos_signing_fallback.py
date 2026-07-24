"""Regression coverage for the macOS desktop signing fallback in install.sh."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_macos_signing_block() -> str:
    text = INSTALL_SH.read_text()
    match = re.search(
        r'    # macOS: use the same config-aware signing fixup.*?^    fi\n',
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, "macOS signing block not found"
    return match.group(0)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_executable_python_failure_runs_historical_signing_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CSC_LINK", "inherited-publisher-config")
    monkeypatch.setenv("APPLE_SIGNING_IDENTITY", "inherited-publisher-identity")
    install_dir = tmp_path / "hermes-agent"
    python = install_dir / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    _write_executable(python, "#!/bin/sh\nexit 42\n")

    desktop_dir = install_dir / "apps" / "desktop"
    app = desktop_dir / "release" / "mac-arm64" / "Hermes.app"
    app.mkdir(parents=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls.log"
    for command in ("xattr", "codesign"):
        _write_executable(
            fake_bin / command,
            f'#!/bin/sh\nprintf "%s\\n" "{command} $*" >> "$CALLS"\n',
        )

    script = f"""
set -e
log_warn() {{ printf 'WARN: %s\\n' "$*"; }}
run_signing_block() {{
{_extract_macos_signing_block()}
}}
run_signing_block
"""
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "CALLS": str(calls),
        "OS": "macos",
        "INSTALL_DIR": str(install_dir),
        "HERMES_HOME": str(tmp_path / "home"),
        "desktop_dir": str(desktop_dir),
        "app": str(app),
    }
    env.pop("CSC_LINK", None)
    env.pop("APPLE_SIGNING_IDENTITY", None)
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "config-aware macos signing fixup failed" in result.stdout.lower()
    assert calls.read_text().splitlines() == [
        f"xattr -cr {app}",
        f"codesign --force --deep --sign - {app}",
    ]


def test_python_helper_keeps_shell_publisher_decision_after_dotenv_load(
    tmp_path: Path,
) -> None:
    install_dir = tmp_path / "hermes-agent"
    python = install_dir / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    _write_executable(python, '#!/bin/sh\nexec "$TEST_PYTHON" "$@"\n')

    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text("CSC_LINK=dotenv-publisher-config\n")

    desktop_dir = install_dir / "apps" / "desktop"
    app = desktop_dir / "release" / "mac-arm64" / "Hermes.app"
    executable = app / "Contents" / "MacOS" / "Hermes"
    executable.parent.mkdir(parents=True)
    executable.touch()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls.log"
    for command in ("xattr", "codesign"):
        _write_executable(
            fake_bin / command,
            f'#!/bin/sh\nprintf "%s\\n" "{command} $*" >> "$CALLS"\n',
        )

    script = f"""
set -e
log_warn() {{ printf 'WARN: %s\\n' "$*"; }}
run_signing_block() {{
{_extract_macos_signing_block()}
}}
run_signing_block
"""
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "PYTHONPATH": str(REPO_ROOT),
        "CALLS": str(calls),
        "TEST_PYTHON": sys.executable,
        "OS": "macos",
        "INSTALL_DIR": str(install_dir),
        "HERMES_HOME": str(hermes_home),
        "desktop_dir": str(desktop_dir),
        "app": str(app),
    }
    env.pop("CSC_LINK", None)
    env.pop("APPLE_SIGNING_IDENTITY", None)
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "config-aware macos signing fixup failed" not in result.stdout.lower()
    assert calls.read_text().splitlines() == [
        f"xattr -cr {app}",
        f"codesign --force --deep --sign - {app}",
        f"codesign --verify --deep --strict {app}",
    ]
