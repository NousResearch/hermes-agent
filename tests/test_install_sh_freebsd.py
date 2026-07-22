"""Regression tests for the FreeBSD branches in install.sh.

FreeBSD has no astral uv binary, no nodejs.org tarballs, and no Playwright
browser builds — the installer sources those from pkg instead. The critical
invariant is that the managed-uv contract still holds: whatever the source,
the binary Hermes uses lives at ``$HERMES_HOME/bin/uv``, the same path
``hermes_cli/managed_uv.py`` resolves at runtime.

Behavioral tests source the relevant functions from install.sh in a stubbed
shell (same harness style as test_install_sh_browser_install.py).
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


# ---------------------------------------------------------------------------
# Text assertions
# ---------------------------------------------------------------------------

def test_detect_os_recognizes_freebsd() -> None:
    text = INSTALL_SH.read_text()
    assert 'FreeBSD*)' in text
    assert 'OS="freebsd"' in text
    assert 'DISTRO="freebsd"' in text


def test_freebsd_browser_hint_uses_the_env_var_hermes_reads() -> None:
    """The runtime consumes AGENT_BROWSER_EXECUTABLE_PATH (tools/browser_tool.py).

    An earlier draft documented PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH, which
    Hermes never reads — following that instruction configures nothing.
    """
    text = INSTALL_SH.read_text()
    assert "export AGENT_BROWSER_EXECUTABLE_PATH=/usr/local/bin/chromium" in text
    assert "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH" not in text


def test_freebsd_system_packages_resolve_to_pkg() -> None:
    text = INSTALL_SH.read_text()
    assert 'freebsd)       pkg_install="pkg install -y"   ;;' in text


def test_freebsd_uv_branch_preserves_managed_location() -> None:
    """uv may come from pkg, but it must be linked into $HERMES_HOME/bin/uv."""
    text = INSTALL_SH.read_text()
    assert 'ln -sf "$_sys_uv" "$_managed_uv"' in text
    # Failure mode is actionable, not a dead astral.sh pointer.
    assert "sudo pkg install uv" in text


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------

def _extract_functions(*names: str) -> str:
    src = INSTALL_SH.read_text()
    out = []
    for name in names:
        m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", src,
                      re.MULTILINE | re.DOTALL)
        assert m, f"could not extract {name}() from install.sh"
        # Drop full-line comments: install.sh's prose (e.g. "the runtime
        # update path ... hermes update") would otherwise trip conftest's
        # live-system guard, which scans the subprocess command string for
        # hermes-update-shaped token sequences.
        body = "\n".join(
            line for line in m.group(0).splitlines()
            if not line.lstrip().startswith("#")
        )
        out.append(body)
    return "\n\n".join(out)


def _run_harness(body: str, script: str, env: dict | None = None) -> subprocess.CompletedProcess:
    harness = f"""
log_info() {{ echo "INFO: $*"; }}
log_success() {{ echo "SUCCESS: $*"; }}
log_warn() {{ echo "WARN: $*"; }}
log_error() {{ echo "ERROR: $*" >&2; }}

{body}

{script}
"""
    return subprocess.run(["bash", "-c", harness], capture_output=True,
                          text=True, env={**os.environ, **(env or {})})


def test_detect_os_sets_freebsd_vars() -> None:
    body = _extract_functions("detect_os")
    proc = _run_harness(body, """
is_termux() { return 1; }
uname() { echo "FreeBSD"; }
detect_os
echo "RESULT OS=$OS DISTRO=$DISTRO"
""")
    assert "RESULT OS=freebsd DISTRO=freebsd" in proc.stdout


def _install_uv_harness(tmp: Path, *, uv_on_path: bool, pkg_works: bool,
                        as_root: bool) -> subprocess.CompletedProcess:
    """Drive install_uv() with OS=freebsd in a controlled PATH sandbox.

    ``uv_on_path`` pre-creates a fake system uv; ``pkg_works`` makes the pkg
    stub "install" one onto PATH; ``as_root`` controls the id(1) stub. sudo is
    stubbed to always fail, so the non-root pkg path is never taken. PATH is
    pinned to the sandbox bin plus /usr/bin:/bin (for mkdir/ln) so a real uv
    on the developer's machine can never satisfy `command -v uv`.
    """
    sysbin = tmp / "sysbin"
    sysbin.mkdir()
    fake_uv = sysbin / "uv"
    payload = tmp / "uv-payload"
    payload.write_text("#!/bin/sh\necho 'uv 0.7.0 (fake)'\n")
    payload.chmod(0o755)
    if uv_on_path:
        fake_uv.write_bytes(payload.read_bytes())
        fake_uv.chmod(0o755)

    body = _extract_functions("install_uv")
    pkg_stub = (
        f'pkg() {{ echo "pkg $*" >> "$RUNLOG"; cp "{payload}" "{fake_uv}"; '
        f'chmod 755 "{fake_uv}"; }}'
        if pkg_works
        else 'pkg() { echo "pkg $*" >> "$RUNLOG"; return 1; }'
    )
    uid = "0" if as_root else "1000"
    script = f"""
OS=freebsd
DISTRO=freebsd
HERMES_HOME="{tmp}/.hermes"
PATH="{sysbin}:/usr/bin:/bin"
id() {{ echo {uid}; }}
sudo() {{ return 1; }}
{pkg_stub}
install_uv
echo "RESULT UV_CMD=$UV_CMD"
"""
    runlog = tmp / "runlog"
    runlog.touch()
    return _run_harness(body, script, env={"RUNLOG": str(runlog)})


def test_install_uv_links_existing_system_uv_into_hermes_bin(tmp_path) -> None:
    proc = _install_uv_harness(tmp_path, uv_on_path=True, pkg_works=False,
                               as_root=False)
    managed = tmp_path / ".hermes" / "bin" / "uv"
    assert f"RESULT UV_CMD={managed}" in proc.stdout, proc.stdout + proc.stderr
    assert managed.is_symlink()
    assert os.path.realpath(managed) == os.path.realpath(tmp_path / "sysbin" / "uv")
    # No pkg attempt needed when a system uv already exists.
    assert "pkg install" not in (tmp_path / "runlog").read_text()


def test_install_uv_falls_back_to_pkg_as_root(tmp_path) -> None:
    proc = _install_uv_harness(tmp_path, uv_on_path=False, pkg_works=True,
                               as_root=True)
    managed = tmp_path / ".hermes" / "bin" / "uv"
    assert f"RESULT UV_CMD={managed}" in proc.stdout, proc.stdout + proc.stderr
    assert managed.is_symlink()
    assert "pkg install -y uv" in (tmp_path / "runlog").read_text()


def test_install_uv_fails_actionably_without_uv_or_pkg(tmp_path) -> None:
    proc = _install_uv_harness(tmp_path, uv_on_path=False, pkg_works=False,
                               as_root=True)
    assert proc.returncode == 1
    assert "astral.sh ships no FreeBSD binary" in proc.stderr
    assert "sudo pkg install uv" in proc.stdout
    assert not (tmp_path / ".hermes" / "bin" / "uv").exists()


def _install_node_harness(tmp: Path, *, pkg_works: bool) -> subprocess.CompletedProcess:
    body = _extract_functions("install_node")
    pkg_stub = (
        'pkg() { echo "pkg $*" >> "$RUNLOG"; return 0; }'
        if pkg_works
        else 'pkg() { echo "pkg $*" >> "$RUNLOG"; return 1; }'
    )
    node_stub = 'node() { echo "v22.14.0"; }' if pkg_works else ""
    script = f"""
OS=freebsd
DISTRO=freebsd
id() {{ echo 0; }}
{pkg_stub}
{node_stub}
install_node
echo "RESULT HAS_NODE=$HAS_NODE"
"""
    runlog = tmp / "runlog"
    runlog.touch()
    return _run_harness(body, script, env={"RUNLOG": str(runlog)})


def test_install_node_uses_pkg_on_freebsd(tmp_path) -> None:
    proc = _install_node_harness(tmp_path, pkg_works=True)
    assert "RESULT HAS_NODE=true" in proc.stdout, proc.stdout + proc.stderr
    assert "pkg install -y node npm" in (tmp_path / "runlog").read_text()
    # Never falls through to the nodejs.org tarball path (no FreeBSD builds).
    assert "Downloading" not in proc.stdout


def test_install_node_pkg_failure_is_nonfatal_with_hint(tmp_path) -> None:
    proc = _install_node_harness(tmp_path, pkg_works=False)
    assert proc.returncode == 0
    assert "RESULT HAS_NODE=false" in proc.stdout
    assert "sudo pkg install node npm" in proc.stdout
