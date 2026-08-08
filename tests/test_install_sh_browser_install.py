"""Regression tests for install.sh browser setup.

Browser automation is optional. The installer should not leave Hermes
half-installed just because Playwright's managed Chromium download hangs on an
unsupported distribution.

Also covers quoted AGENT_BROWSER_EXECUTABLE_PATH writes: spaced paths, escape
round-trips through python-dotenv and POSIX sh, and Snap override cleanup.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
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


def test_install_script_honors_explicit_browser_override_only() -> None:
    """find_system_browser consults only an explicit AGENT_BROWSER_EXECUTABLE_PATH."""
    text = INSTALL_SH.read_text()

    assert 'override="${AGENT_BROWSER_EXECUTABLE_PATH:-}"' in text
    # An explicit override still skips the bundled download (override, not fallback).
    assert "Skipping bundled Chromium download" in text


def test_playwright_installs_are_timeout_guarded() -> None:
    text = INSTALL_SH.read_text()

    # The timeout wrapper still exists and is used internally by the install
    # wrapper, so every Playwright download remains bounded.
    assert "run_browser_install_with_timeout()" in text
    # Playwright installs now go through run_playwright_install(), which wraps
    # run_browser_install_with_timeout (timeout-guarded) and adds an
    # unrecognized-platform fallback retry.
    assert "run_playwright_install 600 npx playwright install chromium" in text
    # --with-deps is still invoked on apt-based systems, but only when sudo
    # is available non-interactively (root or passwordless sudo). Non-sudo
    # service users fall back to the browser-only install — see
    # install_node_deps() in install.sh.
    assert "run_playwright_install 600 npx playwright install --with-deps chromium" in text
    # The wrapper still bounds the download with the timeout helper.
    assert 'run_browser_install_with_timeout "$timeout_seconds" "$@"' in text


def test_install_script_supports_skip_browser_flag() -> None:
    """--skip-browser (and --no-playwright alias) skips the Playwright install."""
    text = INSTALL_SH.read_text()

    assert "--skip-browser|--no-playwright)" in text
    assert "SKIP_BROWSER=true" in text
    assert 'if [ "$SKIP_BROWSER" = true ]; then' in text
    assert "--skip-browser Skip Playwright/Chromium install" in text


def test_browser_install_timeout_stays_interruptible() -> None:
    """The Playwright download must stay Ctrl+C-able and force-kill if wedged.

    GNU `timeout` runs the child in its own process group, so a terminal Ctrl+C
    reaches `timeout` but never the download — it looks frozen and ignores
    Ctrl+C (#35166). `--foreground` keeps it in the shell's foreground group;
    `-k 10` guarantees a SIGKILL after the deadline. Both are GNU-only, so the
    installer probes support once and falls back to plain `timeout`.
    """
    text = INSTALL_SH.read_text()

    # GNU-flag probe + the guarded invocation must both be present. The timeout
    # binary is parameterized ($timeout_bin) so macOS gtimeout works too (#39219).
    assert '"$timeout_bin" --foreground -k 10 1 true' in text
    assert '"$timeout_bin" --foreground -k 10 "$timeout_seconds" "$@"' in text
    # Plain-timeout fallback preserved for BusyBox/non-GNU.
    assert '"$timeout_bin" "$timeout_seconds" "$@"' in text


# ---------------------------------------------------------------------------
# Behavioral tests: source the install.sh helpers in a stubbed shell and assert
# the override retry fires ONLY on a too-new apt release (#35166), and not on a
# host Playwright already supports.
# ---------------------------------------------------------------------------


def _run_install_fn(distro: str, version: str, *, native_fails: bool,
                    arch: str = "x86_64", operator_override: str = "") -> dict:
    """Source the relevant functions from install.sh and drive run_playwright_install.

    Stubs `npx` (the install command) to fail/succeed, `uname -m` for arch, and
    `log_warn`/`log_info` to no-ops. Returns parsed observations: how many times
    the install command ran, and the override value seen on each run.
    """
    # Extract the functions we need so we don't execute the whole installer.
    # run_browser_install_with_timeout delegates to run_with_timeout (#39219),
    # so the helper must be pulled in too or the install command never runs.
    fn_names = [
        "run_browser_install_with_timeout",
        "run_with_timeout",
        "playwright_host_unrecognized",
        "playwright_fallback_platform",
        "run_playwright_install",
    ]
    src = INSTALL_SH.read_text()

    extracted = []
    for name in fn_names:
        m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", src, re.MULTILINE | re.DOTALL)
        assert m, f"could not extract {name}() from install.sh"
        extracted.append(m.group(0))
    body = "\n\n".join(extracted)

    native_rc = 1 if native_fails else 0
    harness = f"""
set -u
DISTRO={distro!r}
DISTRO_VERSION={version!r}
export PLAYWRIGHT_HOST_PLATFORM_OVERRIDE={operator_override!r}
[ -z "$PLAYWRIGHT_HOST_PLATFORM_OVERRIDE" ] && unset PLAYWRIGHT_HOST_PLATFORM_OVERRIDE

log_warn() {{ :; }}
log_info() {{ :; }}

# Stub `uname -m` for arch control without touching the real binary.
uname() {{ if [ "$1" = "-m" ]; then echo {arch!r}; else command uname "$@"; fi }}

# Stub `timeout`: just run the command, ignoring flags/duration. We only care
# about how the npx stub behaves, not real timeout semantics here.
timeout() {{
    while [ $# -gt 0 ]; do
        case "$1" in -*|[0-9]*) shift ;; *) break ;; esac
    done
    "$@"
}}

# Stub the install command. Record each invocation + the override in effect.
npx() {{
    echo "RUN override=${{PLAYWRIGHT_HOST_PLATFORM_OVERRIDE:-<none>}}" >>"$RUNLOG"
    # First run reflects native_fails; the override retry (if any) succeeds.
    if [ -n "${{PLAYWRIGHT_HOST_PLATFORM_OVERRIDE:-}}" ]; then return 0; fi
    return {native_rc}
}}

{body}

run_playwright_install 600 npx playwright install --with-deps chromium
echo "FINAL_RC=$?"
"""
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as lf:
        runlog = lf.name
    try:
        env = dict(os.environ, RUNLOG=runlog)
        proc = subprocess.run(["bash", "-c", harness], capture_output=True,
                              text=True, env=env)
        runs = Path(runlog).read_text().strip().splitlines()
        final_rc = None
        for line in proc.stdout.splitlines():
            if line.startswith("FINAL_RC="):
                final_rc = int(line.split("=", 1)[1])
        return {"runs": runs, "final_rc": final_rc, "stderr": proc.stderr}
    finally:
        Path(runlog).unlink(missing_ok=True)


def test_override_retry_fires_on_ubuntu_26() -> None:
    """Ubuntu 26.04 (too new) → native fails → retry with ubuntu24.04 override."""
    r = _run_install_fn("ubuntu", "26.04", native_fails=True)
    assert len(r["runs"]) == 2, r["runs"]
    assert "override=<none>" in r["runs"][0]
    assert "override=ubuntu24.04-x64" in r["runs"][1]
    assert r["final_rc"] == 0


def test_override_retry_fires_on_debian_14() -> None:
    """Debian 14 (> 13) is the too-new apt case → retry with override."""
    r = _run_install_fn("debian", "14", native_fails=True)
    assert len(r["runs"]) == 2, r["runs"]
    assert "override=ubuntu24.04-x64" in r["runs"][1]
    assert r["final_rc"] == 0


def test_no_retry_when_native_succeeds_on_ubuntu_26() -> None:
    """Even on Ubuntu 26.04, a successful native install is never retried."""
    r = _run_install_fn("ubuntu", "26.04", native_fails=False)
    assert len(r["runs"]) == 1, r["runs"]
    assert "override=<none>" in r["runs"][0]
    assert r["final_rc"] == 0


# ---------------------------------------------------------------------------
# Quoted AGENT_BROWSER_EXECUTABLE_PATH writes (PR #57249 suite): drive the
# supported installer stages and assert dotenv + POSIX sh round-trips.
# ---------------------------------------------------------------------------


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
