"""Regression tests for Playwright install failure diagnostics (#87340 class).

run_playwright_install's first attempt discarded stderr (2>/dev/null), so a
failed Playwright download surfaced as a bare "failed or hung" with no TLS,
404, or disk-space detail — the same undiagnosable-output class #87340 fixed
for npm. The attempts now capture stderr and print it when they fail; the
download progress bar stays live on stdout.
"""

import re
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

STUBS = """\
log_warn() { echo "warn: $*"; }
log_info() { echo "info: $*"; }
playwright_host_unrecognized() { return 1; }
playwright_fallback_platform() { echo "ubuntu24.04-x64"; }
"""

FAILING_ATTEMPT = """\
run_browser_install_with_timeout() { echo "UNABLE_TO_VERIFY_LEAF_SIGNATURE" >&2; return 1; }
"""

SUCCEEDING_ATTEMPT = """\
run_browser_install_with_timeout() { echo "downloading..."; return 0; }
"""

UNRECOGNIZED_HOST = """\
playwright_host_unrecognized() { return 0; }
"""


def _run_wrapper(
    tmp_path: Path, stubs: str, args: str = "60 npx playwright install chromium"
) -> subprocess.CompletedProcess:
    """Execute run_playwright_install from install.sh against stubbed helpers."""
    body = re.search(
        r"^run_playwright_install\(\) \{.*?^\}",
        INSTALL_SH.read_text(),
        re.S | re.M,
    )
    assert body, "run_playwright_install not found in install.sh"

    script = tmp_path / "harness.sh"
    script.write_text(textwrap.dedent(STUBS) + stubs + body.group(0)
                      + f"\nrun_playwright_install {args}\n")
    return subprocess.run(["bash", str(script)], capture_output=True, text=True)


def test_failed_attempt_prints_captured_stderr(tmp_path: Path) -> None:
    """stderr must reach the user on failure — /dev/null hid TLS/404/disk causes."""
    result = _run_wrapper(tmp_path, FAILING_ATTEMPT)
    assert result.returncode == 1
    assert "Playwright install output:" in result.stdout
    assert "UNABLE_TO_VERIFY_LEAF_SIGNATURE" in result.stderr


def test_successful_attempt_stays_quiet(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path, SUCCEEDING_ATTEMPT)
    assert result.returncode == 0
    assert "Playwright install output:" not in result.stdout


def test_override_retry_also_prints_captured_stderr(tmp_path: Path) -> None:
    """The platform-override retry must surface its output too."""
    result = _run_wrapper(
        tmp_path, UNRECOGNIZED_HOST + FAILING_ATTEMPT
    )
    assert result.returncode == 1
    assert "PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64" in result.stdout
    assert "Playwright install output:" in result.stdout
    assert "Playwright retry output:" in result.stdout
    assert result.stderr.count("UNABLE_TO_VERIFY_LEAF_SIGNATURE") == 2
