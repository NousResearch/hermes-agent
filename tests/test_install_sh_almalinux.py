"""Regression tests for AlmaLinux 8 install support (#98938).

RHEL 8-family systems (AlmaLinux/Rocky/CentOS/RHEL 8) ship GCC 8.5, which
predates the -std=gnu++20 flag name node-gyp passes when building node-pty —
the build fails even though a compiler is technically present. Separately,
`/etc/os-release`'s `ID` on AlmaLinux is the literal string `almalinux`, not
`alma`, so the RPM-family case patterns in install.sh must match both.
"""

import re
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_function_body(source: str, name: str) -> str:
    m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", source, re.MULTILINE | re.DOTALL)
    assert m, f"could not extract {name}() from install.sh"
    return m.group(0)


def test_playwright_deps_branch_matches_almalinux() -> None:
    text = INSTALL_SH.read_text()
    assert "fedora|rhel|centos|rocky|alma|almalinux)" in text


def test_check_cxx_compiler_has_rhel_family_gcc_toolset_branch() -> None:
    text = INSTALL_SH.read_text()

    body = _extract_function_body(text, "check_cxx_compiler")
    assert "rhel|centos|rocky|alma|almalinux" in body
    assert "gcc-toolset-12" in body


def _run_cxx_compiler_ok(distro: str, *, gxx_present: bool, gxx_major: str) -> dict:
    """Source cxx_compiler_ok() from install.sh with a stubbed g++ and run it."""
    src = INSTALL_SH.read_text()
    body = _extract_function_body(src, "cxx_compiler_ok")

    if gxx_present:
        gxx_stub = f"""
g++() {{
    if [ "$1" = "-dumpversion" ]; then echo {gxx_major!r}; else :; fi
}}
"""
    else:
        gxx_stub = ""

    harness = f"""
set -u
DISTRO={distro!r}
command() {{
    if [ "$1" = "-v" ] && [ "$2" = "g++" ]; then
        {"return 0" if gxx_present else "return 1"}
    fi
    if [ "$1" = "-v" ] && [ "$2" = "clang++" ]; then
        return 1
    fi
    builtin command "$@"
}}

{gxx_stub}

{body}

cxx_compiler_ok
echo "RC=$?"
"""
    proc = subprocess.run(["bash", "-c", harness], capture_output=True, text=True)
    rc = None
    for line in proc.stdout.splitlines():
        if line.startswith("RC="):
            rc = int(line.split("=", 1)[1])
    return {"rc": rc, "stderr": proc.stderr}


def test_almalinux_with_old_gcc_is_not_ok() -> None:
    """AlmaLinux 8's default GCC 8 must be treated as insufficient."""
    r = _run_cxx_compiler_ok("almalinux", gxx_present=True, gxx_major="8")
    assert r["rc"] == 1, r


def test_almalinux_with_gcc_toolset_12_is_ok() -> None:
    """Once gcc-toolset-12 (GCC 12) is on PATH, the check must pass."""
    r = _run_cxx_compiler_ok("almalinux", gxx_present=True, gxx_major="12")
    assert r["rc"] == 0, r


def test_ubuntu_only_checks_presence_not_version() -> None:
    """Non-RHEL distros keep the old presence-only behavior (no regression)."""
    r = _run_cxx_compiler_ok("ubuntu", gxx_present=True, gxx_major="8")
    assert r["rc"] == 0, r
