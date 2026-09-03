"""Regression tests for the Hermes-managed Node's npm global prefix.

When the installer falls back to a bundled Node under ``$HERMES_HOME/node``,
npm's default global prefix is that Node dir, so ``npm install -g <pkg>``
drops the package binary in ``$HERMES_HOME/node/bin`` — which is NOT on PATH
(only the command link dir is) and is wiped on every Node upgrade. Users then
report "I can ``npm i -g`` but the package isn't usable on the command line".

The fix redirects the bundled Node's global prefix to the command link dir's
parent (so global bins land in the already-on-PATH link dir alongside
node/npm/npx), scoped to the bundled Node via its prefix-local global npmrc.
"""

from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
NODE_BOOTSTRAP = REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh"


def test_install_sh_redirects_bundled_npm_global_prefix_to_link_dir() -> None:
    text = INSTALL_SH.read_text()

    # The redirect must target the link dir's PARENT so global bins resolve to
    # <parent>/bin == the command link dir (node/npm/npx live there and it is
    # guaranteed on PATH by the installer's PATH setup).
    assert "configure_managed_node_npm_prefix()" in text
    assert 'printf \'prefix=%s\\n\' "$(dirname "$link_dir")" > "$HERMES_HOME/node/etc/npmrc"' in text


def test_install_sh_repairs_existing_managed_node_on_rerun() -> None:
    """The redirect must run on every install (not just fresh Node installs),
    so re-running the installer repairs pre-existing managed installs whose
    Node is already up to date and would otherwise skip install_node."""
    text = INSTALL_SH.read_text()

    check_node_body = text.split("check_node()", 1)[1].split("\ninstall_node()", 1)[0]
    assert "configure_managed_node_npm_prefix" in check_node_body

    # No-op guard so it's safe to call when there is no managed Node.
    assert '[ -x "$HERMES_HOME/node/bin/npm" ] || return 0' in text


def test_node_bootstrap_redirects_bundled_npm_global_prefix_to_link_dir() -> None:
    text = NODE_BOOTSTRAP.read_text()

    assert "_nb_configure_npm_prefix()" in text
    assert 'printf \'prefix=%s\\n\' "$(dirname "$_link_dir")" > "$HERMES_HOME/node/etc/npmrc"' in text

    # Runs at the top of ensure_node so existing managed installs are repaired
    # even when a modern Node is already present (early return path).
    ensure_node_body = text.split("ensure_node()", 1)[1]
    assert "_nb_configure_npm_prefix" in ensure_node_body
    assert '[ -x "$HERMES_HOME/node/bin/npm" ] || return 0' in text
    assert "heal_managed_node()" in text
    assert "_nb_managed_tool_broken" in text
    assert "for tool in node npm npx" in text


def test_prefix_redirect_is_gated_on_user_managed_node_detection() -> None:
    """The PATH redirect must be kept for managed-only users AND scoped to the
    managed tree when a user-managed runtime exists. The fix must not destroy
    the feature it secures (globals on PATH for managed-only users)."""
    for path, detector in (
        (INSTALL_SH, "is_user_managed_node"),
        (NODE_BOOTSTRAP, "_nb_is_user_managed_node"),
    ):
        text = path.read_text()
        assert f"{detector}()" in text
        assert f"if {detector}; then" in text
        # User-managed branch: scope to the managed tree, never claim the
        # user's prefix. (The managed-only link-dir redirect is pinned by the
        # original tests above.)
        assert 'printf \'prefix=%s\\n\' "$HERMES_HOME/node" > "$HERMES_HOME/node/etc/npmrc"' in text


def test_detection_skips_managed_tree_path_entries_and_symlinks() -> None:
    """Detection must ignore PATH entries inside $HERMES_HOME/node and symlinks
    that resolve back into it (the installer links npm into the command link
    dir — that must not count as a user runtime)."""
    for path in (INSTALL_SH, NODE_BOOTSTRAP):
        text = path.read_text()
        assert '"$HERMES_HOME/node"|"$HERMES_HOME/node"/*) continue ;;' in text
        assert 'while [ -L "$' in text
        assert 'readlink' in text
    install_text = INSTALL_SH.read_text()
    assert 'for entry in "$dir/node" "$dir/npm"; do' in install_text
    bootstrap_text = NODE_BOOTSTRAP.read_text()
    assert 'for _entry in "$_dir/node" "$_dir/npm"; do' in bootstrap_text


def _make_sysbin(tmp_path: Path) -> Path:
    """A hermetic bin dir of wrapper scripts for the external tools the
    installer functions shell out to. Wrappers, not copied binaries: macOS
    kills relocated system binaries (Killed: 9), and wrappers keep the case
    PATH free of any real Node."""
    sysbin = tmp_path / "sysbin"
    sysbin.mkdir(exist_ok=True)
    for tool in ("mkdir", "cat", "readlink", "dirname", "basename"):
        src = shutil.which(tool)
        assert src, f"{tool} not found on PATH"
        (sysbin / tool).write_text(f"#!/bin/sh\nexec {shlex.quote(src)} \"$@\"\n")
        (sysbin / tool).chmod(0o755)
    return sysbin


def _prepare_bootstrap_case(tmp_path: Path, with_user_node: bool):
    """Build a hermetic HERMES_HOME + PATH and return (hermes_home, link_dir,
    case_path). Only the stub tools are on PATH, so a system Node on the CI
    runner cannot contaminate the managed-only case."""
    hermes_home = tmp_path / "hermes-home"
    node_bin = hermes_home / "node" / "bin"
    node_bin.mkdir(parents=True)
    for tool in ("node", "npm", "npx"):
        (node_bin / tool).write_text("#!/bin/sh\nexit 0\n")
        (node_bin / tool).chmod(0o755)

    link_dir = tmp_path / "fake-local" / "bin"
    link_dir.mkdir(parents=True)

    sysbin = _make_sysbin(tmp_path)

    path_entries = []
    if with_user_node:
        user_bin = tmp_path / "user-node" / "bin"
        user_bin.mkdir(parents=True)
        for tool in ("node", "npm"):
            (user_bin / tool).write_text("#!/bin/sh\nexit 0\n")
            (user_bin / tool).chmod(0o755)
        path_entries.append(str(user_bin))
    else:
        # Hermes symlinks npm into the command link dir; the symlink must not
        # be mistaken for a user runtime.
        (link_dir / "npm").symlink_to(node_bin / "npm")
        path_entries.append(str(link_dir))

    path_entries.append(str(sysbin))
    return hermes_home, link_dir, ":".join(path_entries)


def _run_bootstrap_case(tmp_path: Path, with_user_node: bool) -> str:
    hermes_home, link_dir, case_path = _prepare_bootstrap_case(tmp_path, with_user_node)
    script = (
        "set -euo pipefail\n"
        f"source {shlex.quote(str(NODE_BOOTSTRAP))}\n"
        f"_nb_get_link_dir() {{ echo {shlex.quote(str(link_dir))}; }}\n"
        "_nb_configure_npm_prefix\n"
        'cat "$HERMES_HOME/node/etc/npmrc"\n'
    )
    bash_bin = shutil.which("bash")
    assert bash_bin, "bash not found"
    proc = subprocess.run(
        [bash_bin, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={
            "PATH": case_path,
            "HERMES_HOME": str(hermes_home),
            "HOME": str(tmp_path),
            "LANG": "C",
        },
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash installer")
def test_node_bootstrap_scopes_prefix_to_managed_tree_with_user_node(tmp_path) -> None:
    """With a user-managed Node on PATH, the managed npm must scope to its own
    tree and never claim the user's prefix."""
    out = _run_bootstrap_case(tmp_path, with_user_node=True)
    assert out.strip() == f"prefix={tmp_path}/hermes-home/node"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash installer")
def test_node_bootstrap_redirects_to_link_dir_without_user_node(tmp_path) -> None:
    """With no user-managed Node (only the managed tree's own symlink), the
    original link-dir redirect is preserved."""
    out = _run_bootstrap_case(tmp_path, with_user_node=False)
    assert out.strip() == f"prefix={tmp_path}/fake-local"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash installer")
def test_node_bootstrap_chained_relative_symlinks_into_managed_tree(tmp_path) -> None:
    """A chain of relative symlinks (with `..` hops) that lands inside the
    managed tree must not be mistaken for a user runtime.

    Regression for the per-hop canonicalization fix: relative link targets
    resolve against the current link's directory, and `..` segments are
    normalized, so the chain's true physical target is compared against
    $HERMES_HOME/node.
    """
    hermes_home = tmp_path / "hermes-home"
    node_bin = hermes_home / "node" / "bin"
    node_bin.mkdir(parents=True)
    for tool in ("node", "npm", "npx"):
        (node_bin / tool).write_text("#!/bin/sh\nexit 0\n")
        (node_bin / tool).chmod(0o755)

    link_bin = tmp_path / "fake-local" / "bin"
    link_bin.mkdir(parents=True)
    chain_dir = tmp_path / "chain"
    chain_dir.mkdir()
    # First hop: relative link out of the link dir (fake-local/bin ->
    # tmp/chain is two levels up); second hop: relative link (with ..
    # segments) into the managed tree.
    (link_bin / "npm").symlink_to(Path("..") / ".." / "chain" / "npm")
    (chain_dir / "npm").symlink_to(os.path.relpath(node_bin / "npm", chain_dir))

    sysbin = _make_sysbin(tmp_path)
    script = (
        "set -euo pipefail\n"
        f"source {shlex.quote(str(NODE_BOOTSTRAP))}\n"
        f"_nb_get_link_dir() {{ echo {shlex.quote(str(link_bin))}; }}\n"
        "_nb_configure_npm_prefix\n"
        'cat "$HERMES_HOME/node/etc/npmrc"\n'
    )
    bash_bin = shutil.which("bash")
    assert bash_bin, "bash not found"
    proc = subprocess.run(
        [bash_bin, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={
            "PATH": f"{link_bin}:{sysbin}",
            "HERMES_HOME": str(hermes_home),
            "HOME": str(tmp_path),
            "LANG": "C",
        },
    )
    assert proc.returncode == 0, proc.stderr
    # The chain resolves into the managed tree → not a user runtime → the
    # original link-dir redirect is preserved.
    assert proc.stdout.strip() == f"prefix={tmp_path}/fake-local"
