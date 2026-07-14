"""Regression tests for broken-Node.js recovery in install.sh (issue #38636).

The installer's ``check_node`` runs an execution probe: a ``node`` binary that
is present but whose ``node --version`` exits non-zero (e.g. linked against a
newer libc/OpenSSL than is installed) is treated as *unusable* rather than
merely "too old". These tests exercise that path end-to-end and, in
particular, the Termux/Android recovery branch:

* Android is classified ``OS="android"`` and the bundled Node tarball server
  only ships ``linux``/``macos`` builds, so on Termux the *only* viable
  remediation is ``pkg`` — even for a broken binary. Forcing the managed
  (bundled) route there dead-ends at "Unsupported OS" with ``HAS_NODE=false``.
* A zero ``pkg`` exit is not proof that ``node`` runs. The installer must
  re-probe after the package command and only report ``HAS_NODE=true`` when a
  supported version actually runs.

Rather than assert on script text, each test extracts the real
``check_node`` / ``install_node`` (and their helpers) from ``install.sh`` and
runs them under bash with stubbed ``node`` / ``pkg`` / ``curl`` so behaviour —
not wording — is what's pinned.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

# Functions pulled verbatim from install.sh and sourced into the harness.
_EXTRACTED = (
    "node_satisfies_build",
    "node_version_looks_valid",
    "node_probe_ok",
    "check_node",
    "install_node",
)


def _extract_function(name: str, text: str) -> str:
    """Return ``<name>() { ... }`` verbatim (signature + body + closing brace).

    Anchored on ``^<name>() {`` and the next top-of-line ``}`` so it keeps
    working when neighbouring functions move.
    """
    match = re.search(
        rf"^{re.escape(name)}\(\)\s*\{{\n.*?^\}}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"{name}() not found in scripts/install.sh"
    return match.group(0)


def _run_check_node(
    tmp_path: Path,
    *,
    distro: str,
    os_id: str,
    node_present: bool,
    node_broken: bool,
    pkg_fixes_node: bool = False,
    pkg_exit: int = 0,
) -> tuple[str, str]:
    """Run the real ``check_node`` under a stubbed environment.

    Returns ``(has_node, pkg_log)`` where ``has_node`` is the final
    ``HAS_NODE`` value ("true"/"false") and ``pkg_log`` is the newline-joined
    argv of every ``pkg`` invocation (empty string if ``pkg`` was never run).
    """
    text = INSTALL_SH.read_text()
    functions = "\n\n".join(_extract_function(name, text) for name in _EXTRACTED)

    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    fixed_flag = tmp_path / "node_fixed"  # created by pkg stub when it "repairs" node
    pkg_log = tmp_path / "pkg.log"
    hermes_home = tmp_path / "hermes_home"  # deliberately has no managed node/bin/node

    # ``node`` stub: present-but-broken until the pkg stub drops the fixed flag.
    if node_present:
        (stub_bin / "node").write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            f'  if [ -f "{fixed_flag}" ]; then echo "v22.14.0"; exit 0; fi\n'
            + (
                '  echo "node: error while loading shared libraries" 1>&2; exit 1\n'
                if node_broken
                else '  echo "v18.0.0"; exit 0\n'  # runs, but below the build floor
            )
            + "fi\nexit 0\n"
        )
        (stub_bin / "node").chmod(0o755)

    # ``pkg`` stub: log argv, optionally repair node, exit with pkg_exit.
    (stub_bin / "pkg").write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{pkg_log}"\n'
        + (f': > "{fixed_flag}"\n' if pkg_fixes_node else "")
        + f"exit {pkg_exit}\n"
    )
    (stub_bin / "pkg").chmod(0o755)

    # ``curl`` stub: never touch the network. Empty output makes the bundled
    # downloader resolve no tarball and bail with HAS_NODE=false.
    (stub_bin / "curl").write_text("#!/bin/sh\nexit 0\n")
    (stub_bin / "curl").chmod(0o755)

    driver = f"""#!/usr/bin/env bash
export PATH="{stub_bin}:$PATH"
HERMES_HOME="{hermes_home}"
NODE_VERSION=22
OS="{os_id}"
DISTRO="{distro}"
HAS_NODE=false
log_info() {{ :; }}
log_warn() {{ :; }}
log_success() {{ :; }}
log_error() {{ :; }}

{functions}

check_node || true
echo "HAS_NODE=${{HAS_NODE}}"
"""
    result = subprocess.run(
        ["bash", "-c", driver],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": os.environ.get("PATH", "")},
    )
    assert result.returncode == 0, f"harness failed: {result.stderr}"
    m = re.search(r"^HAS_NODE=(\w+)$", result.stdout, re.MULTILINE)
    assert m is not None, f"no HAS_NODE line in output: {result.stdout!r}"
    log_text = pkg_log.read_text() if pkg_log.exists() else ""
    return m.group(1), log_text


def test_install_sh_defines_node_probe_ok() -> None:
    """The execution-probe helper must exist (used for the post-pkg re-probe)."""
    assert "node_probe_ok()" in INSTALL_SH.read_text()


def test_termux_broken_node_repaired_via_pkg_reinstall(tmp_path: Path) -> None:
    """Broken system Node on Termux: pkg *reinstall* repairs it, then re-probe passes."""
    has_node, pkg_log = _run_check_node(
        tmp_path,
        distro="termux",
        os_id="android",
        node_present=True,
        node_broken=True,
        pkg_fixes_node=True,
    )
    assert has_node == "true"
    # A no-op ``pkg install`` cannot repair an already-present broken package;
    # the recovery must use reinstall.
    assert "reinstall" in pkg_log


def test_termux_broken_node_not_repaired_reports_unavailable(tmp_path: Path) -> None:
    """The core regression: pkg exits 0 but node still won't run → HAS_NODE stays false.

    The pre-fix branch set ``HAS_NODE=true`` off the ``pkg`` exit code alone,
    without re-probing, and would have declared a still-broken Node available.
    """
    has_node, pkg_log = _run_check_node(
        tmp_path,
        distro="termux",
        os_id="android",
        node_present=True,
        node_broken=True,
        pkg_fixes_node=False,  # pkg "succeeds" but node is still broken
    )
    assert has_node == "false"
    assert "reinstall" in pkg_log


def test_termux_broken_node_pkg_command_fails(tmp_path: Path) -> None:
    """When the pkg command itself fails, Node is reported unavailable, not present."""
    has_node, _ = _run_check_node(
        tmp_path,
        distro="termux",
        os_id="android",
        node_present=True,
        node_broken=True,
        pkg_fixes_node=False,
        pkg_exit=1,
    )
    assert has_node == "false"


def test_non_termux_broken_node_never_uses_pkg(tmp_path: Path) -> None:
    """On a bundled-capable platform a broken Node routes to the managed download.

    The Termux ``pkg`` remediation must not fire on linux — the bundled
    fallback is reserved for platforms the dist server actually serves. Here
    the stubbed ``curl`` resolves no tarball, so the managed install bails with
    ``HAS_NODE=false`` and, crucially, ``pkg`` is never invoked.
    """
    has_node, pkg_log = _run_check_node(
        tmp_path,
        distro="ubuntu",
        os_id="linux",
        node_present=True,
        node_broken=True,
    )
    assert has_node == "false"
    assert pkg_log == ""
