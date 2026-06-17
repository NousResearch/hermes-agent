"""Regression tests for Hermes-managed Node PATH isolation.

When the installer falls back to a bundled Node under ``$HERMES_HOME/node``,
Hermes must keep node/npm/npx private. Older installers linked them into the
command link dir (usually ``~/.local/bin``), which could shadow nvm/fnm/Volta.
The managed npm prefix now stays inside ``$HERMES_HOME/node`` so npm global
bins land in ``$HERMES_HOME/node/bin``: private to Hermes, but still visible to
Hermes subprocess PATH construction and dependency detection. Startup migrates
only legacy symlinks that still point into the current Hermes-managed Node.
"""

import os
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
NODE_BOOTSTRAP = REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh"


def test_install_sh_keeps_bundled_npm_global_prefix_inside_hermes_home() -> None:
    text = INSTALL_SH.read_text()

    # Hermes-managed npm must stay self-contained while still placing global
    # package bins in a Hermes runtime PATH directory. $HERMES_HOME/node/bin is
    # already used by install.sh --prefix installs and dep_ensure detection.
    assert "configure_managed_node_npm_prefix()" in text
    assert 'printf \'prefix=%s\\n\' "$HERMES_HOME/node" > "$HERMES_HOME/node/etc/npmrc"' in text
    assert 'printf \'prefix=%s\\n\' "$(dirname "$link_dir")" > "$HERMES_HOME/node/etc/npmrc"' not in text


def test_managed_npm_global_bin_is_a_hermes_runtime_detection_path(tmp_path) -> None:
    """prefix=$HERMES_HOME/node makes npm -g bins visible to Hermes internals."""
    from hermes_cli.dep_ensure import _has_hermes_agent_browser

    managed_prefix = tmp_path / "node"
    managed_global_bin = managed_prefix / "bin"
    assert managed_global_bin == tmp_path / "node" / "bin"

    managed_global_bin.mkdir(parents=True)
    agent_browser = managed_global_bin / "agent-browser"
    agent_browser.write_text("#!/bin/sh\n", encoding="utf-8")
    agent_browser.chmod(0o755)

    with patch("hermes_cli.dep_ensure._IS_WINDOWS", False), \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        assert _has_hermes_agent_browser() is True


def test_install_sh_repairs_existing_managed_node_on_rerun() -> None:
    """The redirect must run on every install (not just fresh Node installs),
    so re-running the installer repairs pre-existing managed installs whose
    Node is already up to date and would otherwise skip install_node."""
    text = INSTALL_SH.read_text()

    check_node_body = text.split("check_node()", 1)[1].split("\ninstall_node()", 1)[0]
    assert "configure_managed_node_npm_prefix" in check_node_body

    # No-op guard so it's safe to call when there is no managed Node.
    assert '[ -x "$HERMES_HOME/node/bin/npm" ] || return 0' in text


def test_node_bootstrap_keeps_bundled_npm_global_prefix_inside_hermes_home() -> None:
    text = NODE_BOOTSTRAP.read_text()

    assert "_nb_configure_npm_prefix()" in text
    assert 'printf \'prefix=%s\\n\' "$HERMES_HOME/node" > "$HERMES_HOME/node/etc/npmrc"' in text
    assert 'printf \'prefix=%s\\n\' "$(dirname "$_link_dir")" > "$HERMES_HOME/node/etc/npmrc"' not in text

    # Runs at the top of ensure_node so existing managed installs are repaired
    # even when a modern Node is already present (early return path).
    ensure_node_body = text.split("ensure_node()", 1)[1]
    assert "_nb_configure_npm_prefix" in ensure_node_body
    assert '[ -x "$HERMES_HOME/node/bin/npm" ] || return 0' in text


def test_installers_never_link_managed_node_tools_into_command_dir() -> None:
    install_text = INSTALL_SH.read_text()
    bootstrap_text = NODE_BOOTSTRAP.read_text()

    forbidden = (
        'ln -sf "$HERMES_HOME/node/bin/node"',
        'ln -sf "$HERMES_HOME/node/bin/npm"',
        'ln -sf "$HERMES_HOME/node/bin/npx"',
    )
    for snippet in forbidden:
        assert snippet not in install_text
        assert snippet not in bootstrap_text


def test_node_bootstrap_migrates_only_hermes_owned_node_symlinks(tmp_path) -> None:
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    local_bin = home / ".local" / "bin"
    node_bin = hermes_home / "node" / "bin"
    nvm_bin = home / ".nvm" / "versions" / "node" / "v22.12.0" / "bin"
    node_bin.mkdir(parents=True)
    local_bin.mkdir(parents=True)
    nvm_bin.mkdir(parents=True)

    for name in ("node", "npm", "npx"):
        target = node_bin / name
        target.write_text("#!/bin/sh\n", encoding="utf-8")
        target.chmod(0o755)
        (local_bin / name).symlink_to(target)

    user_node = nvm_bin / "node"
    user_node.write_text("#!/bin/sh\n", encoding="utf-8")
    user_node.chmod(0o755)
    user_link = local_bin / "user-node"
    user_link.symlink_to(user_node)

    script = textwrap.dedent(
        f"""
        source "{NODE_BOOTSTRAP}"
        _nb_remove_legacy_node_links
        for name in node npm npx; do
            [ ! -e "{local_bin}/$name" ] && [ ! -L "{local_bin}/$name" ] || exit 10
        done
        [ -L "{user_link}" ] || exit 11
        """
    )
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script],
        env={
            "HOME": str(home),
            "HERMES_HOME": str(hermes_home),
            "PATH": os.environ.get("PATH", ""),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_node_bootstrap_prefers_existing_nvm_node_over_private_fallback(tmp_path) -> None:
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    hermes_node_bin = hermes_home / "node" / "bin"
    nvm_node_bin = home / ".nvm" / "versions" / "node" / "v22.20.0" / "bin"
    hermes_node_bin.mkdir(parents=True)
    nvm_node_bin.mkdir(parents=True)

    for name in ("node", "npm"):
        binary = hermes_node_bin / name
        binary.write_text("#!/bin/sh\necho v22.12.0\n", encoding="utf-8")
        binary.chmod(0o755)

    nvm_node = nvm_node_bin / "node"
    nvm_node.write_text("#!/bin/sh\necho v22.20.0\n", encoding="utf-8")
    nvm_node.chmod(0o755)

    script = textwrap.dedent(
        f"""
        source "{NODE_BOOTSTRAP}"
        ensure_node >/tmp/hermes-bootstrap-prefers-nvm.out 2>/tmp/hermes-bootstrap-prefers-nvm.err
        [ "$(command -v node)" = "{nvm_node}" ] || {{
            command -v node
            cat /tmp/hermes-bootstrap-prefers-nvm.err >&2
            exit 12
        }}
        """
    )
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script],
        env={
            "HOME": str(home),
            "HERMES_HOME": str(hermes_home),
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_install_sh_checks_user_version_managers_before_private_node() -> None:
    text = INSTALL_SH.read_text()
    check_node_body = text.split("check_node()", 1)[1].split("\ninstall_node()", 1)[0]

    fnm_idx = check_node_body.index("try_existing_fnm_node")
    volta_idx = check_node_body.index("try_existing_volta_node")
    nvm_idx = check_node_body.index("try_existing_nvm_node")
    private_idx = check_node_body.index('activate_node_bin_if_satisfies_build "$HERMES_HOME/node/bin/node"')

    assert fnm_idx < private_idx
    assert volta_idx < private_idx
    assert nvm_idx < private_idx
    assert '"${FNM_DIR:-$HOME/.local/share/fnm}"' in text
    assert '${VOLTA_HOME:-$HOME/.volta}' in text
    assert '${NVM_DIR:-$HOME/.nvm}' in text
