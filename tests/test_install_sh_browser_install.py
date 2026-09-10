"""Regression tests for install.sh browser setup.

Browser automation is optional. The installer should not leave Hermes
half-installed just because Playwright's managed Chromium download hangs on an
unsupported distribution.
"""

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"




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
    # bounded IPv4 retry plus the unrecognized-platform fallback.
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
# every failed download gets one IPv4-only retry, while the platform override
# remains limited to too-new apt releases (#35166).
# ---------------------------------------------------------------------------

def _run_install_fn(
    distro: str,
    version: str,
    *,
    native_fails: bool,
    fallback_fails: bool = False,
    arch: str = "x86_64",
    operator_override: str = "",
    operator_proxy: str = "",
    operator_all_proxy: str = "",
) -> dict:
    """Source the relevant functions from install.sh and drive run_playwright_install.

    Stubs `npx` (the install command) to fail/succeed, `uname -m` for arch, and
    `log_warn`/`log_info` to no-ops. Returns parsed observations: how many times
    the install command ran, the override value, and whether IPv4 was forced.
    """
    # Extract the functions we need so we don't execute the whole installer.
    # run_browser_install_with_timeout delegates to run_with_timeout (#39219),
    # so the helper must be pulled in too or the install command never runs.
    fn_names = [
        "run_browser_install_with_timeout",
        "run_with_timeout",
        "playwright_host_unrecognized",
        "playwright_fallback_platform",
        "run_playwright_install_ipv4",
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
    fallback_rc = 1 if fallback_fails else 0
    harness = f"""
set -u
DISTRO={distro!r}
DISTRO_VERSION={version!r}
INSTALL_DIR={str(REPO_ROOT)!r}
export PLAYWRIGHT_HOST_PLATFORM_OVERRIDE={operator_override!r}
[ -z "$PLAYWRIGHT_HOST_PLATFORM_OVERRIDE" ] && unset PLAYWRIGHT_HOST_PLATFORM_OVERRIDE
export HTTPS_PROXY={operator_proxy!r}
[ -z "$HTTPS_PROXY" ] && unset HTTPS_PROXY
unset https_proxy
export ALL_PROXY={operator_all_proxy!r}
[ -z "$ALL_PROXY" ] && unset ALL_PROXY
unset all_proxy

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

# Stub the install command. Record each invocation, the platform override, and
# whether the retry routed through the temporary IPv4-only CONNECT proxy.
npx() {{
    case "${{HTTPS_PROXY:-}}" in
        http://hermes:*@127.0.0.1:*) ipv4=yes ;;
        *) ipv4=no ;;
    esac
    echo "RUN override=${{PLAYWRIGHT_HOST_PLATFORM_OVERRIDE:-<none>}} ipv4=$ipv4 proxy=${{HTTPS_PROXY:-<none>}}" >>"$RUNLOG"
    # The normal and fallback outcomes are controlled independently.
    if [ "$ipv4" = yes ]; then return {fallback_rc}; fi
    return {native_rc}
}}

{body}

run_playwright_install 600 npx playwright install --with-deps chromium
echo "FINAL_RC=$?"
"""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as lf:
        runlog = lf.name
    try:
        env = dict(os.environ, RUNLOG=runlog)
        proc = subprocess.run(
            ["bash", "-c", harness], capture_output=True, text=True, env=env
        )
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


def test_failed_supported_host_retries_once_with_forced_ipv4() -> None:
    """A download failure on a supported host gets one real IPv4-only retry."""
    r = _run_install_fn("arch", "rolling", native_fails=True)

    assert len(r["runs"]) == 2, r["runs"]
    assert "ipv4=no" in r["runs"][0]
    assert "override=<none>" in r["runs"][1]
    assert "ipv4=yes" in r["runs"][1]
    assert "proxy=http://hermes:" in r["runs"][1]
    assert "@127.0.0.1:" in r["runs"][1]
    assert r["final_rc"] == 0


def test_platform_fallback_retry_is_also_forced_to_ipv4() -> None:
    """Too-new apt hosts combine the existing platform and network fallbacks."""
    r = _run_install_fn("ubuntu", "26.04", native_fails=True)

    assert len(r["runs"]) == 2, r["runs"]
    assert "override=ubuntu24.04-x64" in r["runs"][1]
    assert "ipv4=yes" in r["runs"][1]
    assert r["final_rc"] == 0


def test_operator_proxy_failure_is_not_bypassed() -> None:
    """A failed operator proxy is authoritative and disables the local relay."""
    r = _run_install_fn(
        "arch",
        "rolling",
        native_fails=True,
        operator_proxy="https://operator-proxy.invalid:8443",
    )

    assert len(r["runs"]) == 1, r["runs"]
    assert "proxy=https://operator-proxy.invalid:8443" in r["runs"][0]
    assert r["final_rc"] == 1


def test_operator_all_proxy_failure_is_not_bypassed() -> None:
    """ALL_PROXY is also an authoritative operator egress policy."""
    r = _run_install_fn(
        "arch",
        "rolling",
        native_fails=True,
        operator_all_proxy="socks5://operator-proxy.invalid:1080",
    )

    assert len(r["runs"]) == 1, r["runs"]
    assert "ipv4=no" in r["runs"][0]
    assert r["final_rc"] == 1


def test_ipv4_retry_runs_once_then_surfaces_failure() -> None:
    """If both attempts fail, stop after the single retry and return failure."""
    r = _run_install_fn(
        "arch", "rolling", native_fails=True, fallback_fails=True
    )

    assert len(r["runs"]) == 2, r["runs"]
    assert "ipv4=no" in r["runs"][0]
    assert "ipv4=yes" in r["runs"][1]
    assert r["final_rc"] == 1


def test_operator_platform_override_is_preserved_for_both_attempts() -> None:
    """An explicit platform pin remains intact during the IPv4 retry."""
    override = "debian13-x64"
    r = _run_install_fn(
        "debian",
        "13",
        native_fails=True,
        operator_override=override,
    )

    assert len(r["runs"]) == 2, r["runs"]
    assert f"override={override}" in r["runs"][0]
    assert f"override={override}" in r["runs"][1]
    assert "ipv4=no" in r["runs"][0]
    assert "ipv4=yes" in r["runs"][1]
    assert r["final_rc"] == 0


import re


def _extract_function_body(source: str, name: str) -> str:
    m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", source, re.MULTILINE | re.DOTALL)
    assert m, f"could not extract {name}() from install.sh"
    return m.group(0)


def test_ensure_browser_no_longer_npm_installs_agent_browser() -> None:
    """agent-browser resolves lazily via npx everywhere else in the system
    (tools/browser_tool.py::_find_agent_browser); this was the last place
    that still eagerly npm-installed a second, separately version-pinned
    copy of it. Removed: agent-browser acquisition now happens only via
    `hermes update`'s npx cache warm or an actual browser-tool call's lazy
    npx resolution (PR #44772 review)."""
    body = _extract_function_body(INSTALL_SH.read_text(), "ensure_browser")

    assert "agent-browser@" not in body
    assert "Installing Chromium via agent-browser install" not in body
    # camofox is unrelated to this change and must still be installed here.
    assert "@askjo/camofox-browser@^1.5.2" in body
    # System-browser detection is still cheap/valuable without agent-browser.
    assert "find_system_browser" in body
    assert "configure_browser_env_from_system_browser" in body


def test_ensure_browser_still_ignore_scripts_and_timeout_guarded() -> None:
    """The removal of agent-browser must not have also dropped the
    supply-chain and hang-protection hardening that still applies to the
    remaining camofox install."""
    body = _extract_function_body(INSTALL_SH.read_text(), "ensure_browser")

    assert "--ignore-scripts" in body
    assert "run_with_timeout" in body


def test_ensure_browser_no_longer_references_agent_browser_binary_path() -> None:
    """No dangling reference to a local agent-browser binary path should
    remain now that this function never installs it — a leftover reference
    would be dead code pointing at a binary that no longer gets placed
    there by this function."""
    body = _extract_function_body(INSTALL_SH.read_text(), "ensure_browser")

    assert "$HERMES_HOME/node/bin/agent-browser" not in body





