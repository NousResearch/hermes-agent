"""The installer must never adopt a pre-release Node.js build.

`nodejs.org/dist/latest-v26.x/` serves `node-v26.8.0-<os>-<arch>.tar.xz` — a
filename that looks like a final release — whose binary reports
`v26.8.0-alpha.0.0.0`. Node publishes a headers tarball only for final
releases, so `process.release.headersUrl` 404s for that build and node-gyp
cannot compile against it.

That is fatal rather than cosmetic: `node-pty` ships prebuilds for darwin and
win32 only, so every Linux install compiles it, and the failure surfaces as an
opaque `node-pty@1.1.0 install: command failed` because the package's
`node scripts/prebuild.js || node-gyp rebuild` fallback swallows gyp's stderr.

These tests drive the real blocks out of `scripts/install.sh` rather than
asserting on a copy, so the guard cannot drift away from the test.
"""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
NODE_BOOTSTRAP = REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh"


def _extract_function(text: str, name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{\n.*?^\}}\n", text, re.M | re.S
    )
    assert match, (
        f"could not locate {name}() in scripts/install.sh — "
        "if it was renamed, update this test with it"
    )
    return match.group(0)


def _node_satisfies_build(version: str) -> bool:
    """Run the real node_satisfies_build() against one `node --version` string."""
    script = (
        _extract_function(INSTALL_SH.read_text(encoding="utf-8"), "node_satisfies_build")
        + f'\nnode_satisfies_build "{version}"\n'
    )
    return subprocess.run(["bash", "-c", script]).returncode == 0


def test_prerelease_node_never_satisfies_the_build_floor() -> None:
    """The exact build that breaks node-pty, plus the other pre-release tags."""
    assert not _node_satisfies_build("v26.8.0-alpha.0.0.0")
    assert not _node_satisfies_build("v24.0.0-rc.1")
    assert not _node_satisfies_build("v23.5.0-nightly20260101abcdef")
    assert not _node_satisfies_build("v22.22.0-pre")


def test_final_releases_still_satisfy_the_build_floor() -> None:
    """The guard must not cost us any version that actually works."""
    assert _node_satisfies_build("v22.22.0")  # the floor itself
    assert _node_satisfies_build("v24.20.0")
    assert _node_satisfies_build("v26.8.0")  # a real 26 release, once one ships


def test_versions_below_the_floor_are_still_rejected() -> None:
    assert not _node_satisfies_build("v22.21.0")
    assert not _node_satisfies_build("v20.19.0")
    assert not _node_satisfies_build("not-a-version")


def test_install_node_falls_back_to_an_older_release_line(tmp_path: Path) -> None:
    """A rejected line must not end the install — the next one is tried.

    install_node_line() cannot be exercised here (it downloads), so it is
    stubbed to reject 26 the way a pre-release tarball does and accept 24.
    """
    body = _extract_function(INSTALL_SH.read_text(encoding="utf-8"), "install_node")
    script = f"""
set -e
NODE_VERSION=26
DISTRO=linux
OS=linux
uname() {{ echo x86_64; }}
log_info(){{ :; }}; log_warn(){{ :; }}; log_success(){{ :; }}
install_node_line() {{
    echo "$1" >> "$TRIED"
    [ "$1" = 24 ]
}}
{body}
install_node
"""
    tried = tmp_path / "tried"
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={"PATH": "/usr/bin:/bin", "TRIED": str(tried)},
    )
    assert result.returncode == 0, result.stderr
    # 26 first, then the previous even (LTS) line; stops once one works.
    assert tried.read_text(encoding="utf-8").split() == ["26", "24"]


def test_downloaded_tree_is_probed_before_it_replaces_anything() -> None:
    """The filename is not evidence — the extracted binary must be checked.

    Guarded as a text assertion because the surrounding block downloads; the
    fallback behaviour it feeds is covered functionally above.
    """
    text = INSTALL_SH.read_text(encoding="utf-8")
    probe = text.index('candidate_ver=$("$extracted_dir/bin/node" --version 2>/dev/null)')
    reject = text.index('if ! node_satisfies_build "$candidate_ver"; then', probe)
    # The probe has to run before the tree is moved into place, or a bad build
    # has already clobbered a working managed Node.
    assert reject < text.index('mv "$extracted_dir" "$HERMES_HOME/node"', probe)


def test_node_bootstrap_mirrors_the_prerelease_guard() -> None:
    """The sourceable helper answers the same question and must agree."""
    text = NODE_BOOTSTRAP.read_text(encoding="utf-8")
    assert "_nb_node_is_prerelease" in text
    assert "_nb_node_is_prerelease && return 1" in text

