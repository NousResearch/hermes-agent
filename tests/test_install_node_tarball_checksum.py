"""Node.js tarball checksum verification in the POSIX installer (#106027).

Every binary-bearing channel of ``scripts/install.sh`` was checksum-pinned
except the Node.js tarball: ``install_node_line`` resolved the name from the
nodejs.org index page, downloaded it, and extracted/executed it with nothing
but the TLS certificate standing between a tampered artifact and the user's
``~/.hermes/node``. (uv: hardcoded sha256; Python: ``uv.lock``; npm:
``package-lock.json`` integrity; Electron: ``SHASUMS256.txt``; git clone:
object hashes.)

The contract pinned here:
- After the tarball download, ``SHASUMS256.txt`` is fetched from the same
  index URL and the tarball's sha256 must match its entry before anything is
  extracted.
- Fail closed on every error path: SHASUMS256.txt unfetchable, no entry for
  this tarball, a hash mismatch, or no hash tool on the box all reject the
  release line (``return 1``, the caller then tries an older line) and leave
  ``$HERMES_HOME/node`` untouched.
- A matching checksum still adopts the tree (the existing flow is unchanged
  when verification passes).

The behavioral tests extract the real ``install_node_line`` from the
installer and run it under a stubbed environment (fake ``curl`` serving
local fixtures; ``log_*``/``node_satisfies_build``/link helpers stubbed), so
the verification logic is exercised end-to-end without network access.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None
    or shutil.which("tar") is None
    or (shutil.which("shasum") is None and shutil.which("sha256sum") is None),
    reason="needs bash, tar, and a sha256 tool",
)

TARBALL_NAME = "node-v26.8.1-darwin-arm64.tar.gz"


def _install_node_line_source() -> str:
    """The ``install_node_line`` function body straight from the installer."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    match = re.search(
        r"^install_node_line\(\) \{.*?^\}", text, re.DOTALL | re.MULTILINE
    )
    assert match is not None, "install_node_line() not found in install.sh"
    return match.group(0)


def _write_stub_curl(bin_dir: Path) -> None:
    (bin_dir / "curl").write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            # stub curl: [-fsSL] <url> [-o <file>] serving fixtures from $CURL_ROOT
            out=""; url=""
            args=("$@")
            for ((i=0; i<${#args[@]}; i++)); do
              case "${args[$i]}" in
                -o) out="${args[$((i+1))]}"; ((i++)) ;;
                -*) ;;
                *) url="${args[$i]}" ;;
              esac
            done
            case "$url" in
              *SHASUMS256.txt)
                [ "${SHASUMS_UNFETCHABLE:-0}" = 1 ] && exit 22
                src="$CURL_ROOT/shasums.txt" ;;
              *latest-v26.x/) src="$CURL_ROOT/index.html" ;;
              *.tar.gz) src="$CURL_ROOT/$TARBALL_NAME" ;;
              *) echo "stub curl: no route for $url" >&2; exit 22 ;;
            esac
            if [ -n "$out" ]; then cat "$src" > "$out"; else cat "$src"; fi
            """
        ),
        encoding="utf-8",
    )
    (bin_dir / "curl").chmod(0o755)


def _build_fixtures(root: Path, shasums_mode: str) -> None:
    """Index page + real tar.gz carrying an executable ``bin/node`` + SHASUMS."""
    stage = root / "stage"
    node_bin_dir = stage / "node-v26.8.1-darwin-arm64" / "bin"
    node_bin_dir.mkdir(parents=True)
    node_bin = node_bin_dir / "node"
    node_bin.write_text("#!/bin/sh\necho v26.8.1\n", encoding="utf-8")
    node_bin.chmod(0o755)
    subprocess.run(
        [
            "tar",
            "czf",
            str(root / TARBALL_NAME),
            "-C",
            str(stage),
            "node-v26.8.1-darwin-arm64",
        ],
        check=True,
    )
    shutil.rmtree(stage)

    (root / "index.html").write_text(
        f'<a href="{TARBALL_NAME}">{TARBALL_NAME}</a>\n', encoding="utf-8"
    )

    digest = hashlib.sha256((root / TARBALL_NAME).read_bytes()).hexdigest()
    if shasums_mode == "good":
        shasums = f"{digest}  {TARBALL_NAME}\n"
    elif shasums_mode == "mismatch":
        shasums = f"{'a' * 64}  {TARBALL_NAME}\n"
    elif shasums_mode == "no-entry":
        shasums = f"{digest}  some-other-artifact.tar.gz\n"
    else:
        raise ValueError(shasums_mode)
    (root / "shasums.txt").write_text(shasums, encoding="utf-8")


def _run_install_node_line(
    tmp_path: Path, *, shasums_unfetchable: bool = False
) -> tuple[int, Path]:
    """Run the extracted ``install_node_line`` under the stubbed environment.

    Returns ``(return code, hermes home)``; the caller asserts on the tree.
    """
    fixtures = tmp_path / "fix"
    bin_dir = tmp_path / "bin"
    hermes_home = tmp_path / "hermes_home"
    binlinks = tmp_path / "binlinks"
    for directory in (fixtures, bin_dir, hermes_home, binlinks):
        directory.mkdir(parents=True, exist_ok=True)
    _write_stub_curl(bin_dir)

    script = tmp_path / "run_case.sh"
    script.write_text(
        textwrap.dedent(
            f"""\
            set -u
            export CURL_ROOT="{fixtures}"
            export TARBALL_NAME="{TARBALL_NAME}"
            export SHASUMS_UNFETCHABLE={"1" if shasums_unfetchable else "0"}
            export PATH="{bin_dir}:$PATH"
            log_info() {{ :; }}
            log_warn() {{ :; }}
            log_error() {{ :; }}
            log_success() {{ :; }}
            node_satisfies_build() {{ return 0; }}
            get_command_link_dir() {{ echo "{binlinks}"; }}
            configure_managed_node_npm_prefix() {{ :; }}
            OS=darwin
            DISTRO=none
            HERMES_HOME="{hermes_home}"
            HAS_NODE=false
            export HERMES_HOME HAS_NODE OS DISTRO
            install_node_line 26 darwin arm64
            rc=$?
            echo "RC=$rc"
            echo "HAS_NODE=$HAS_NODE"
            exit $rc
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "func.sh").write_text(_install_node_line_source(), encoding="utf-8")
    # The function must be sourced into the same shell that runs the case.
    result = subprocess.run(
        ["bash", "-c", f'source "{tmp_path / "func.sh"}"; source "{script}"'],
        capture_output=True,
        text=True,
        timeout=120,
    )
    tail = result.stdout.strip().splitlines()[-2:] if result.stdout.strip() else []
    rc = 1
    for line in tail:
        if line.startswith("RC="):
            rc = int(line.split("=", 1)[1])
    return rc, hermes_home


@pytest.mark.parametrize(
    "mode,unfetchable,expected_rc,expected_tree",
    [
        ("good", False, 0, True),
        ("mismatch", False, 1, False),
        ("no-entry", False, 1, False),
        ("good", True, 1, False),  # SHASUMS256.txt unfetchable: fail closed
    ],
    ids=[
        "match-adopts",
        "mismatch-refuses",
        "missing-entry-refuses",
        "shasums-unfetchable-refuses",
    ],
)
def test_install_node_line_checksum_gate(
    tmp_path, mode, unfetchable, expected_rc, expected_tree
):
    _build_fixtures(tmp_path / "fix", mode)
    (tmp_path / "fix").mkdir(exist_ok=True)
    rc, hermes_home = _run_install_node_line(tmp_path, shasums_unfetchable=unfetchable)
    assert rc == expected_rc, f"install_node_line must exit {expected_rc} (mode={mode})"
    adopted = (hermes_home / "node" / "bin" / "node").exists()
    assert adopted is expected_tree, (
        "a verified tarball must be adopted, and every failure path must leave "
        "$HERMES_HOME/node untouched"
    )


def test_download_is_verified_before_extraction():
    """Structural pin: the SHASUMS256.txt check sits between download and extract."""
    source = _install_node_line_source()
    download = source.index("Downloading $tarball_name")
    shasums = source.index("SHASUMS256.txt")
    extract = source.index("Extracting to")
    assert download < shasums < extract, (
        "the checksum gate must run after the tarball download and before extraction"
    )


def test_checksum_gate_fails_closed():
    """Structural pin: every error path rejects the line instead of continuing."""
    source = _install_node_line_source()
    gate = source[source.index("Verify the tarball") : source.index("Extracting to")]
    for refusal in (
        'rm -rf "$tmp_dir"',
        "return 1",
    ):
        assert refusal in gate, (
            f"checksum gate must clean up and reject ({refusal!r} missing)"
        )
    # The mismatch comparison must cover the no-entry case too: an empty
    # expected hash can never equal a real digest.
    assert '[ -z "$expected" ] || [ "$expected" != "$actual" ]' in gate
    # Both mainstream hash tools are supported so macOS (shasum) and
    # Linux/Termux (sha256sum) installs verify.
    assert "command -v sha256sum" in gate and "shasum -a 256" in gate
