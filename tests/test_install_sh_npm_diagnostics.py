"""Regression tests for install.sh npm failure diagnostics (#87340, #38016).

Both install-blocking `npm install` calls captured output for diagnosis while
still passing --silent, which suppresses npm's own error reporting. The capture
recorded an empty log, so a failed install printed "npm output:" with nothing
behind it. Node also reads neither SSL_CERT_FILE nor CURL_CA_BUNDLE, so a
corporate MITM proxy surfaces as an opaque failure on POSIX; install.ps1 has
had a hint for this since #38016 and install.sh had none.
"""

import re
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

# A real npm failure through an intercepting proxy, trimmed to the error lines.
CERT_FAILURE = """\
npm warn Unknown env config "nodedir".
npm error code UNABLE_TO_VERIFY_LEAF_SIGNATURE
npm error errno UNABLE_TO_VERIFY_LEAF_SIGNATURE
npm error request to https://registry.npmjs.org/undici/-/undici-6.28.0.tgz \
failed, reason: unable to verify the first certificate
"""

UNRELATED_FAILURE = """\
npm error code E404
npm error 404 Not Found - GET https://registry.npmjs.org/does-not-exist
"""


def _install_blocking_npm_calls() -> list[str]:
    """The `npm install` invocations whose failure aborts the install."""
    return re.findall(
        r"run_with_timeout \"\$NODE_DEPS_TIMEOUT\" npm install[^\n]*",
        INSTALL_SH.read_text(),
    )


def test_install_blocking_npm_calls_are_not_silenced() -> None:
    """--silent defeats the output capture it sits next to (#87340)."""
    calls = _install_blocking_npm_calls()
    assert calls, "expected to find the install-blocking npm install calls"
    offenders = [call for call in calls if "--silent" in call]
    assert not offenders, (
        "`npm install --silent` suppresses npm's error output, so the capture "
        "records an empty log and the installer prints a bare 'npm output:'. "
        "Offending calls: " + repr(offenders)
    )


def test_cert_hint_is_wired_into_both_failure_paths() -> None:
    """Every install-blocking npm failure path must offer the hint."""
    text = INSTALL_SH.read_text()
    assert text.count("npm_cert_hint ") >= 2, (
        "both the node-deps and TUI npm failure branches should call "
        "npm_cert_hint"
    )


def _run_hint(tmp_path: Path, npm_output: str) -> subprocess.CompletedProcess:
    """Execute npm_cert_hint against a captured npm log."""
    body = re.search(
        r"^npm_cert_hint\(\) \{.*?^\}", INSTALL_SH.read_text(), re.S | re.M
    )
    assert body, "npm_cert_hint not found in install.sh"

    log = tmp_path / "npm.log"
    log.write_text(npm_output)
    script = tmp_path / "harness.sh"
    script.write_text(
        textwrap.dedent(
            """\
            log_warn() { echo "warn: $*"; }
            log_info() { echo "info: $*"; }
            """
        )
        + body.group(0)
        + f'\nnpm_cert_hint "{log}"\n'
    )
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True
    )


def test_hint_fires_on_a_real_tls_failure(tmp_path: Path) -> None:
    result = _run_hint(tmp_path, CERT_FAILURE)
    assert result.returncode == 0
    assert "TLS certificate-trust failure" in result.stdout
    # Node needs its own variable; pointing curl at the CA is not enough.
    assert "NODE_EXTRA_CA_CERTS" in result.stdout
    assert "--use-system-ca" in result.stdout


def test_hint_stays_quiet_on_unrelated_failures(tmp_path: Path) -> None:
    result = _run_hint(tmp_path, UNRELATED_FAILURE)
    assert result.returncode == 1
    assert result.stdout == ""


def test_hint_stays_quiet_on_empty_output(tmp_path: Path) -> None:
    result = _run_hint(tmp_path, "")
    assert result.returncode == 1
    assert result.stdout == ""
