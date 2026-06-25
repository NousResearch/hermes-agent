"""Regression tests for install.sh browser setup.

Browser automation is optional. The installer should not leave Hermes
half-installed just because Playwright's managed Chromium download hangs on an
unsupported distribution.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_install_script_does_not_autodetect_system_browser_on_path() -> None:
    """The installer must not scan PATH/well-known locations for a browser.

    Auto-detection silently bound the install to whatever ``command -v
    chromium`` resolved to — most damagingly a Snap Chromium, whose sandbox
    blocks agent-browser's control socket and hangs every browser_navigate. The
    fallback was dropped in favor of always using the bundled Playwright
    Chromium, so the old PATH-scan and "use the system browser" path are gone.
    """
    text = INSTALL_SH.read_text()

    assert "find_system_browser()" in text
    assert "google-chrome google-chrome-stable chromium chromium-browser chrome" not in text
    assert "Skipping Playwright browser download; Hermes will use the system browser." not in text


def test_install_script_honors_explicit_browser_override_only() -> None:
    """find_system_browser consults only an explicit AGENT_BROWSER_EXECUTABLE_PATH."""
    text = INSTALL_SH.read_text()

    assert 'override="${AGENT_BROWSER_EXECUTABLE_PATH:-}"' in text
    # An explicit override still skips the bundled download (override, not fallback).
    assert "Skipping bundled Chromium download" in text


def test_install_script_strips_stale_snap_browser_override() -> None:
    """Already-affected installs must auto-recover.

    A pre-existing AGENT_BROWSER_EXECUTABLE_PATH pointing at a Snap Chromium is
    the exact value that hangs the browser tool, and the runtime reads it from
    .env — so the installer strips it (and a Snap override is rejected even when
    set explicitly) so the bundled Chromium download runs on update.
    """
    text = INSTALL_SH.read_text()

    assert "strip_snap_browser_override()" in text
    assert "^AGENT_BROWSER_EXECUTABLE_PATH=/snap/" in text
    # Both install paths invoke the migration before resolving a browser.
    assert text.count("strip_snap_browser_override") >= 3
    # A snap path is rejected by find_system_browser itself.
    assert "/snap/*) return 1 ;;" in text


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


def test_install_script_skips_with_deps_when_no_sudo() -> None:
    """Non-sudo users on apt distros must not block on an interactive sudo prompt."""
    text = INSTALL_SH.read_text()

    # The apt branch must gate --with-deps behind a sudo capability check
    # (root or non-interactive sudo), otherwise the installer hangs for
    # service-user installs (systemd accounts, operator users, etc.).
    assert 'if [ "$(id -u)" -eq 0 ] || (command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null); then' in text
    assert "sudo npx playwright install-deps chromium" in text


def test_playwright_install_retries_with_platform_override_on_failure() -> None:
    """Installer must self-correct when Playwright doesn't recognize the host.

    On apt releases newer than Playwright knows (Ubuntu 26.04, Debian 14, future
    distros) `playwright install` hangs/fails (#35166). run_playwright_install
    must retry ONCE with PLAYWRIGHT_HOST_PLATFORM_OVERRIDE pinned to the newest
    known build — but only when the host is one of those too-new apt releases
    (playwright_host_unrecognized), never on a host Playwright already supports
    (which would force a glibc mismatch, microsoft/playwright#35114), and never
    when the operator pinned the value.
    """
    text = INSTALL_SH.read_text()

    assert "run_playwright_install()" in text
    assert "playwright_fallback_platform()" in text
    assert "playwright_host_unrecognized()" in text
    # Fallback target is the newest known build, arch-aware.
    assert 'echo "ubuntu24.04-x64"' in text
    assert 'echo "ubuntu24.04-arm64"' in text
    # Try native first: only retry after the first attempt fails.
    assert 'if run_browser_install_with_timeout "$timeout_seconds" "$@" 2>/dev/null; then' in text
    # Operator-pinned override is respected (retry skipped).
    assert 'if [ -n "${PLAYWRIGHT_HOST_PLATFORM_OVERRIDE:-}" ]; then' in text
    # The retry is gated on the unrecognized-apt-release check, not any failure.
    assert "if ! playwright_host_unrecognized; then" in text
    # The retry actually sets the override for the child process.
    assert 'PLAYWRIGHT_HOST_PLATFORM_OVERRIDE="$fallback" \\' in text


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

import subprocess


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
        "run_with_timeout",
        "run_browser_install_with_timeout",
        "playwright_host_unrecognized",
        "playwright_fallback_platform",
        "run_playwright_install",
    ]
    src = INSTALL_SH.read_text()
    import re

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
    import tempfile, os
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


def test_override_retry_does_not_fire_on_supported_ubuntu() -> None:
    """Ubuntu 24.04 is recognized by Playwright → a failure is surfaced, no override."""
    r = _run_install_fn("ubuntu", "24.04", native_fails=True)
    assert len(r["runs"]) == 1, r["runs"]
    assert "override=<none>" in r["runs"][0]
    assert r["final_rc"] == 1


def test_override_retry_does_not_fire_on_fedora() -> None:
    """Non-apt distro never triggers the override retry, even on failure."""
    r = _run_install_fn("fedora", "42", native_fails=True)
    assert len(r["runs"]) == 1, r["runs"]
    assert r["final_rc"] == 1


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


def test_operator_override_respected_no_second_run() -> None:
    """An operator-pinned override applies to attempt 1; no second run on failure."""
    r = _run_install_fn("ubuntu", "26.04", native_fails=True,
                        operator_override="ubuntu22.04-x64")
    # The override is set, so the npx stub returns 0 on the first run.
    assert len(r["runs"]) == 1, r["runs"]
    assert "override=ubuntu22.04-x64" in r["runs"][0]
    assert r["final_rc"] == 0


def test_override_retry_skipped_on_unsupported_arch() -> None:
    """Ubuntu 26.04 on an arch with no Playwright build → no fallback retry."""
    r = _run_install_fn("ubuntu", "26.04", native_fails=True, arch="riscv64")
    assert len(r["runs"]) == 1, r["runs"]
    assert r["final_rc"] == 1

