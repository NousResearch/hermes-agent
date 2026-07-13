"""Regression tests for the Hermes-managed Node's npm global prefix.

When the installer falls back to a bundled Node under ``$HERMES_HOME/node``,
npm's default global prefix is that Node dir, so ``npm install -g <pkg>``
drops the package binary in ``$HERMES_HOME/node/bin`` — which is NOT on PATH
(only the command link dir is) and is wiped on every Node upgrade. Users then
report "I can ``npm i -g`` but the package isn't usable on the command line".

The fix redirects the bundled Node's global prefix to the command link dir's
parent (so global bins land in the already-on-PATH link dir), scoped to the
bundled Node via its prefix-local global npmrc.
"""

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
NODE_BOOTSTRAP = REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh"


def test_install_sh_redirects_bundled_npm_global_prefix_to_link_dir() -> None:
    text = INSTALL_SH.read_text()

    # The redirect must target the link dir's PARENT so global bins resolve to
    # <parent>/bin == the command link dir, which the installer puts on PATH.
    assert "configure_managed_node_npm_prefix()" in text
    assert (
        'printf \'prefix=%s\\n\' "$(dirname "$link_dir")" > "$HERMES_HOME/node/etc/npmrc"'
        in text
    )


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
    assert (
        'printf \'prefix=%s\\n\' "$(dirname "$_link_dir")" > "$HERMES_HOME/node/etc/npmrc"'
        in text
    )

    # Runs at the top of ensure_node so existing managed installs are repaired
    # even when a modern Node is already present (early return path).
    ensure_node_body = text.split("ensure_node()", 1)[1]
    assert "_nb_configure_npm_prefix" in ensure_node_body
    assert '[ -x "$HERMES_HOME/node/bin/npm" ] || return 0' in text
    assert "heal_managed_node()" in text
    assert "_nb_managed_tool_broken" in text
    assert "for tool in node npm npx" in text


def _run_bootstrap_helper(
    tmp_path: Path, body: str
) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update({"HOME": str(home), "HERMES_HOME": str(home / ".hermes")})
    return subprocess.run(
        ["bash", "-c", f'source "{NODE_BOOTSTRAP}"\n{body}'],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_node_bootstrap_preserves_unrelated_command_link(tmp_path: Path) -> None:
    home = tmp_path / "home"
    existing = home / ".local" / "bin" / "node"
    existing.parent.mkdir(parents=True)
    user_node = home / ".volta" / "bin" / "node"
    user_node.parent.mkdir(parents=True)
    user_node.write_text("user-owned\n")
    existing.symlink_to(user_node)
    target = home / ".hermes" / "node" / "bin" / "node"
    target.parent.mkdir(parents=True)
    target.write_text("managed\n")

    result = _run_bootstrap_helper(
        tmp_path,
        '_nb_link_managed_tool "$HOME/.local/bin" node',
    )

    assert result.returncode == 0
    assert existing.is_symlink()
    assert existing.resolve() == user_node
    assert "Leaving existing" in result.stderr


def test_node_bootstrap_links_missing_command_idempotently(tmp_path: Path) -> None:
    home = tmp_path / "home"
    link = home / ".local" / "bin" / "node"
    link.parent.mkdir(parents=True)
    target = home / ".hermes" / "node" / "bin" / "node"
    target.parent.mkdir(parents=True)
    target.write_text("managed\n")

    result = _run_bootstrap_helper(
        tmp_path,
        '_nb_link_managed_tool "$HOME/.local/bin" node\n'
        '_nb_link_managed_tool "$HOME/.local/bin" node',
    )

    assert result.returncode == 0
    assert link.is_symlink()
    assert link.readlink() == target


def test_node_bootstrap_finds_volta_node_outside_gui_path(tmp_path: Path) -> None:
    home = tmp_path / "home"
    node = home / ".volta" / "bin" / "node"
    node.parent.mkdir(parents=True)
    node.write_text("#!/bin/sh\necho v22.12.0\n")
    node.chmod(0o755)

    result = _run_bootstrap_helper(
        tmp_path,
        "PATH=/usr/bin:/bin\n"
        "_nb_try_known_node\n"
        'printf "resolved=%s\\n" "$(command -v node)"',
    )

    assert result.returncode == 0
    assert f"resolved={node}" in result.stdout


def test_install_sh_uses_collision_safe_links_and_known_node_probe() -> None:
    text = INSTALL_SH.read_text()
    install_node_body = text.split("install_node()", 1)[1].split(
        "\ncheck_network_prerequisites()", 1
    )[0]
    check_node_body = text.split("check_node()", 1)[1].split("\ninstall_node()", 1)[0]

    assert "ln -sf" not in install_node_body
    assert 'link_managed_node_tool "$node_link_dir" node' in install_node_body
    assert "activate_node_from_known_locations" in check_node_body
    assert "Leaving existing $destination unchanged" in text
