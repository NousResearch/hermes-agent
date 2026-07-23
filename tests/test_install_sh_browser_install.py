"""Behavioral regression tests for install.sh browser environment setup."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from dotenv import dotenv_values


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
BROWSER_ENV_KEY = "AGENT_BROWSER_EXECUTABLE_PATH"

ROUND_TRIP_PATHS = [
    pytest.param("browser executable with spaces", id="spaces"),
    pytest.param(r"C:\Users\p\scoop\apps\chrome.exe", id="windows-chrome"),
    pytest.param(r"D:\bin\firefox.exe", id="windows-firefox"),
    pytest.param(r"E:\tools\tbrowser.exe", id="literal-backslash-t"),
    pytest.param("browser's executable", id="apostrophe"),
    pytest.param('browser "beta" executable', id="double-quote"),
]

SHELL_EXPANSION_PATHS = [
    pytest.param("browser-$HOME", id="dollar"),
    pytest.param("browser-`printf shell`", id="backtick"),
]


def _make_executable(
    path: Path,
    body: str = "#!/bin/sh\nexit 0\n",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


def _run_config_stage(
    tmp_path: Path,
    browser_path: str | None,
    *,
    path_prefix: Path | None = None,
) -> Path:
    """Run the supported config stage against isolated install and data dirs."""
    install_dir = tmp_path / "install"
    hermes_home = tmp_path / "home"
    install_dir.mkdir()

    if browser_path is not None:
        _make_executable(install_dir / browser_path)

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["HERMES_INSTALL_DIR"] = str(install_dir)
    env.pop(BROWSER_ENV_KEY, None)
    if browser_path is not None:
        env[BROWSER_ENV_KEY] = browser_path
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}{os.pathsep}{env['PATH']}"

    installer = subprocess.run(
        ["bash", str(INSTALL_SH), "--stage", "config", "--no-skills"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert installer.returncode == 0, (installer.stdout, installer.stderr)
    assert installer.stderr == "", (installer.stdout, installer.stderr)

    env_file = hermes_home / ".env"
    assert env_file.is_file()
    return env_file


def _run_node_deps_snap_cleanup(tmp_path: Path, quote: str) -> Path:
    """Run snap cleanup through node-deps with all Node commands stubbed."""
    install_dir = tmp_path / "install"
    hermes_home = tmp_path / "home"
    fake_bin = tmp_path / "fake-bin"
    install_dir.mkdir()
    hermes_home.mkdir()
    (install_dir / "package.json").write_text("{}\n")

    _make_executable(
        fake_bin / "node",
        "#!/bin/sh\nprintf 'v22.12.0\\n'\n",
    )
    _make_executable(fake_bin / "npm")
    _make_executable(fake_bin / "npx")
    _make_executable(
        fake_bin / "timeout",
        """#!/bin/sh
while [ "$#" -gt 0 ]; do
    case "$1" in
        --foreground) shift ;;
        -k) shift 2 ;;
        [0-9]*) shift; break ;;
        *) break ;;
    esac
done
exec "$@"
""",
    )

    env_file = hermes_home / ".env"
    env_file.write_text(
        "# Hermes Agent browser tools — explicit browser override.\n"
        f"{BROWSER_ENV_KEY}={quote}/snap/bin/chromium{quote}\n"
        "KEEP_ME=1\n"
    )

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["HERMES_INSTALL_DIR"] = str(install_dir)
    env[BROWSER_ENV_KEY] = "/snap/bin/chromium"
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    installer = subprocess.run(
        ["bash", str(INSTALL_SH), "--stage", "node-deps"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert installer.returncode == 0, (installer.stdout, installer.stderr)
    assert installer.stderr == "", (installer.stdout, installer.stderr)
    return env_file


def _expected_serialized_line(browser_path: str) -> str:
    escaped = browser_path.replace("\\", "\\\\").replace('"', '\\"')
    return f'{BROWSER_ENV_KEY}="{escaped}"'


def _serialized_browser_line(env_file: Path) -> str:
    return next(
        line
        for line in env_file.read_text().splitlines()
        if line.startswith(f"{BROWSER_ENV_KEY}=")
    )


def _source_with_posix_sh(env_file: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "sh",
            "-c",
            '. "$1"\nprintf \'%s\' "$AGENT_BROWSER_EXECUTABLE_PATH"\n',
            "sh",
            str(env_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("browser_path", ROUND_TRIP_PATHS)
def test_config_stage_browser_path_round_trips_through_dotenv_and_sh(
    tmp_path: Path,
    browser_path: str,
) -> None:
    env_file = _run_config_stage(tmp_path, browser_path)

    assert _serialized_browser_line(env_file) == _expected_serialized_line(browser_path)
    assert dotenv_values(env_file)[BROWSER_ENV_KEY] == browser_path

    sourced = _source_with_posix_sh(env_file)
    assert sourced.returncode == 0, (sourced.stdout, sourced.stderr)
    assert sourced.stderr == "", (sourced.stdout, sourced.stderr)
    assert sourced.stdout == browser_path


@pytest.mark.parametrize("browser_path", SHELL_EXPANSION_PATHS)
def test_dotenv_preserves_literal_shell_expansion_paths(
    tmp_path: Path,
    browser_path: str,
) -> None:
    """The canonical dotenv reader must preserve literal dollar/backtick paths."""
    env_file = _run_config_stage(tmp_path, browser_path)

    assert _serialized_browser_line(env_file) == _expected_serialized_line(browser_path)
    assert dotenv_values(env_file)[BROWSER_ENV_KEY] == browser_path


@pytest.mark.parametrize("browser_path", SHELL_EXPANSION_PATHS)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "The canonical dotenv double-quoted form does not escape POSIX-shell "
        "$ or backtick expansion, so sourcing cannot preserve these filenames."
    ),
)
def test_posix_sh_does_not_round_trip_shell_expansion_paths(
    tmp_path: Path,
    browser_path: str,
) -> None:
    """Document the known shell incompatibility without weakening dotenv coverage."""
    env_file = _run_config_stage(tmp_path, browser_path)

    assert _serialized_browser_line(env_file) == _expected_serialized_line(browser_path)
    assert dotenv_values(env_file)[BROWSER_ENV_KEY] == browser_path

    sourced = _source_with_posix_sh(env_file)
    assert sourced.returncode == 0, (sourced.stdout, sourced.stderr)
    assert sourced.stderr == "", (sourced.stdout, sourced.stderr)
    assert sourced.stdout == browser_path


@pytest.mark.parametrize(
    "quote",
    ["", "'", '"'],
    ids=["unquoted", "single-quoted", "double-quoted"],
)
def test_node_deps_stage_strips_quoted_snap_override(
    tmp_path: Path,
    quote: str,
) -> None:
    env_file = _run_node_deps_snap_cleanup(tmp_path, quote)
    raw_env = env_file.read_text()

    assert BROWSER_ENV_KEY not in dotenv_values(env_file)
    assert "Hermes Agent browser tools" not in raw_env
    assert "KEEP_ME=1" in raw_env


def test_config_stage_does_not_autodetect_browser_from_path(tmp_path: Path) -> None:
    """A PATH browser is ignored unless the operator sets an explicit override."""
    fake_bin = tmp_path / "fake-bin"
    _make_executable(fake_bin / "chromium")

    env_file = _run_config_stage(tmp_path, None, path_prefix=fake_bin)

    assert BROWSER_ENV_KEY not in dotenv_values(env_file)
    assert not any(
        line.startswith(f"{BROWSER_ENV_KEY}=")
        for line in env_file.read_text().splitlines()
    )
