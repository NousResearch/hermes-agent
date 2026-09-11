"""Regression tests for install.sh core-utility preflight (#101164).

The installer implicitly relied on awk/sed/tar (and on xz for extracting
Node .tar.xz tarballs) without proving they exist. On a fresh Fedora WSL
install a missing xz made GNU tar fail with "xz: Cannot exec" through
every Node release line, and a missing awk stalled the uv installer with
no error at all. The installer must fail fast on missing core utilities
with actionable hints, and prefer .tar.gz Node tarballs when xz is
unavailable.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_function(text: str, name: str) -> str:
    start = text.index(f"{name}() {{")
    end = text.index("\n}\n", start) + len("\n}\n")
    return text[start:end]


def test_core_utilities_checked_before_uv_and_git() -> None:
    """check_core_utilities must run before install_uv/check_git.

    check_git parses `git --version` through awk and the uv installer needs
    awk too — without the preflight, a missing awk kills the installer deep
    inside those steps instead of with an actionable message (#101164).
    """
    text = INSTALL_SH.read_text()
    assert "check_core_utilities" in text

    main_body = _extract_function(text, "main")
    assert main_body.index("check_core_utilities") < main_body.index("install_uv")
    assert main_body.index("check_core_utilities") < main_body.index("check_git")

    # The desktop bootstrap prerequisites stage keeps the same ordering.
    prereq = text[text.index("        prerequisites)") : text.index("        repository)")]
    assert prereq.index("check_core_utilities") < prereq.index("install_uv")


def test_core_utilities_cover_awk_sed_tar() -> None:
    text = INSTALL_SH.read_text()
    func = _extract_function(text, "check_core_utilities")
    assert "for util in awk sed tar; do" in func


def _run_check_core_utilities(tmp_path: Path, tools: list[str]) -> subprocess.CompletedProcess:
    """Extract check_core_utilities and run it against a stub PATH.

    Only the utilities named in `tools` resolve; everything else (notably
    awk when absent) is invisible to `command -v`.
    """
    text = INSTALL_SH.read_text()
    func = _extract_function(text, "check_core_utilities")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for util in tools:
        stub = bin_dir / util
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)

    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/bin/bash\n"
        "log_error() { printf 'ERR: %s\\n' \"$1\"; }\n"
        "log_info() { printf 'INFO: %s\\n' \"$1\"; }\n"
        "OS=linux\n"
        "DISTRO=fedora\n"
        + func
        + "\ncheck_core_utilities\n"
        + "echo UNREACHABLE\n"
    )

    env = {"PATH": str(bin_dir), "HOME": str(tmp_path)}
    return subprocess.run(
        # Absolute path: the stub PATH intentionally lacks bash, and exec
        # resolves the command against the *child* env's PATH.
        ["/bin/bash", str(harness)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_missing_awk_fails_fast_with_hint(tmp_path: Path) -> None:
    """A missing awk must abort before any dependent step, with a hint.

    Before the fix the script ran on and stalled inside the uv installer
    (which requires awk) with nothing actionable printed (#101164).
    """
    result = _run_check_core_utilities(tmp_path, tools=["sed", "tar"])

    assert result.returncode != 0
    assert "awk" in result.stdout
    assert "UNREACHABLE" not in result.stdout
    # Fedora gets a concrete dnf command.
    assert "dnf install gawk" in result.stdout


def test_all_core_utilities_present_passes(tmp_path: Path) -> None:
    """With awk/sed/tar all resolvable the preflight is a silent no-op."""
    result = _run_check_core_utilities(tmp_path, tools=["awk", "sed", "tar"])

    assert result.returncode == 0
    assert "UNREACHABLE" in result.stdout


def test_node_tarball_prefers_gz_when_xz_missing() -> None:
    """Without xz, the Node download must resolve .tar.gz before .tar.xz.

    GNU tar shells out to the xz binary for .tar.xz; nodejs.org publishes a
    .tar.gz alongside every .tar.xz, so preferring gz keeps the Node install
    working on stripped-down Linux (fresh Fedora WSL) without requiring the
    user to install xz first (#101164).
    """
    text = INSTALL_SH.read_text()
    func = _extract_function(text, "install_node_line")

    # The gz preference is gated to Linux (macOS bsdtar decompresses xz
    # natively and keeps the smaller xz tarballs).
    assert '[ "$OS" = "linux" ] && ! command -v xz >/dev/null 2>&1' in func

    # Under prefer_gz the .tar.gz resolution must come before the .tar.xz one.
    prefer_gz_start = func.index("prefer_gz=true")
    fallback_marker = func.index("# Fallback to .tar.gz")
    prefer_block = func[prefer_gz_start:fallback_marker]
    assert prefer_block.index(".tar\\.gz") < prefer_block.index(".tar\\.xz")
